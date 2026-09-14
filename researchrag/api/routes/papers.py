"""Paper endpoints: upload/ingest, metadata, structure, pages, figures."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

import fitz  # PyMuPDF
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import Response

from researchrag.api.deps import get_ctx, require_paper, run_job
from researchrag.storage.database import new_id, now_iso

log = logging.getLogger("researchrag.api.routes.papers")
router = APIRouter(tags=["papers"])

UPLOADS_DIR = "uploads"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120] or "document.pdf"


@router.get("/papers")
def list_papers(ctx=Depends(get_ctx)):
    return {"papers": ctx.db.list_papers()}


@router.post("/papers", status_code=202)
async def upload_paper(file: UploadFile, ctx=Depends(get_ctx)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    data = await file.read()
    if len(data) < 1000:
        raise HTTPException(status_code=400, detail="File too small to be a valid PDF")
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 100 MB)")

    paper_id = new_id("paper")
    uploads = ctx.settings.data_dir / UPLOADS_DIR / paper_id
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / _safe_name(file.filename or "document.pdf")
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()

    # content-level dedup: same file already ingested?
    dup = ctx.db.query(
        "SELECT id, status FROM papers WHERE file_sha256 = ?", (sha,)
    )
    if dup and dup[0]["status"] == "ready":
        shutil.rmtree(uploads, ignore_errors=True)
        raise HTTPException(
            status_code=409,
            detail=f"File already ingested as {dup[0]['id']}",
        )

    ctx.db.upsert_paper(
        id=paper_id,
        filename=path.name,
        title=None,
        authors_json="[]",
        abstract=None,
        page_count=0,
        file_sha256=sha,
        size_bytes=len(data),
        source_path=str(path),
        parser=ctx.parser.name,
        status="queued",
        error=None,
        created_at=now_iso(),
    )
    job_id = ctx.db.create_job(paper_id, "ingest")
    run_job(ctx, job_id, _ingest, ctx, paper_id, path)
    return {"paper_id": paper_id, "job_id": job_id, "status": "queued"}


def _ingest(ctx, paper_id: str, path: Path, progress=None) -> dict:
    result = ctx.ingestion.ingest_file(path, progress=progress, paper_id=paper_id)
    ctx.invalidate_retriever(paper_id)
    return result.__dict__


@router.get("/papers/{paper_id}")
def get_paper(paper_id: str, ctx=Depends(get_ctx)):
    paper = require_paper(ctx, paper_id)
    paper["jobs"] = [
        j
        for j in ctx.db.query(
            "SELECT * FROM jobs WHERE paper_id = ? ORDER BY created_at DESC LIMIT 5",
            (paper_id,),
        )
    ]
    return paper


@router.delete("/papers/{paper_id}", status_code=204)
def delete_paper(paper_id: str, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)
    ctx.ingestion.remove_paper(paper_id)
    ctx.invalidate_retriever(paper_id)


@router.get("/papers/{paper_id}/structure")
def paper_structure(paper_id: str, ctx=Depends(get_ctx)):
    """Page → block tree for the viewer and the section navigator."""
    require_paper(ctx, paper_id)
    rows = ctx.db.get_blocks(paper_id)
    pages: dict[int, dict] = {}
    sections: list[dict] = []
    for r in rows:
        page = r["page"]
        p = pages.setdefault(page, {"number": page, "blocks": []})
        b = {
            "id": r["id"],
            "type": r["block_type"],
            "text": r["text"][:400] if r["block_type"] in ("table", "figure") else r["text"][:2000],
            "section": r["section_path_json"] and __import__("json").loads(r["section_path_json"]) or [],
            "section_id": r["section_id"],
            "heading_level": r["heading_level"],
            "caption": r["caption"],
            "has_image": bool(r["image_path"]),
            "bbox": __import__("json").loads(r["bbox_json"]) if r["bbox_json"] else None,
        }
        p["blocks"].append(b)
        if r["block_type"] == "heading":
            path = __import__("json").loads(r["section_path_json"])
            sections.append(
                {
                    "id": r["section_id"],
                    "level": r["heading_level"],
                    "path": path,
                    "title": path[-1] if path else r["text"],
                    "page": page,
                    "block_id": r["id"],
                }
            )
    return {
        "paper_id": paper_id,
        "page_count": len(pages),
        "pages": [pages[k] for k in sorted(pages)],
        "sections": sections,
    }


@router.get("/papers/{paper_id}/chunks")
def paper_chunks(paper_id: str, kind: str | None = None, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)
    chunks = ctx.db.get_chunks(paper_id, kind=kind)
    return {"chunks": chunks}


@router.get("/papers/{paper_id}/pages/{page_number}/image")
def page_image(paper_id: str, page_number: int, dpi: int = 110, ctx=Depends(get_ctx)):
    """Rendered page PNG (cached per paper/page/dpi)."""
    paper = require_paper(ctx, paper_id)
    if page_number < 1 or page_number > paper["page_count"]:
        raise HTTPException(status_code=404, detail="Page not found")
    cache_dir = ctx.settings.artifacts_root / paper_id / "pages"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"p{page_number}_{dpi}.png"
    if not cache.exists():
        try:
            doc = fitz.open(paper["source_path"])
            page = doc[page_number - 1]
            zoom = dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pix.save(str(cache))
            doc.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Page render failed: {e}")
    return Response(content=cache.read_bytes(), media_type="image/png")


@router.get("/papers/{paper_id}/blocks/{block_id}/image")
def block_image(paper_id: str, block_id: str, ctx=Depends(get_ctx)):
    """Figure image extracted during parsing."""
    require_paper(ctx, paper_id)
    rows = ctx.db.query("SELECT image_path FROM blocks WHERE id = ?", (block_id,))
    if not rows or not rows[0]["image_path"]:
        raise HTTPException(status_code=404, detail="No image for this block")
    p = Path(rows[0]["image_path"])
    if not p.exists():
        raise HTTPException(status_code=410, detail="Image file no longer on disk")
    return Response(content=p.read_bytes(), media_type="image/png")
