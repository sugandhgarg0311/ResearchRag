"""Implementation specification endpoints (thin wrappers over spec services)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from researchrag.api.deps import get_ctx, require_paper, run_job
from researchrag.spec.evidence import spec_evidence
from researchrag.spec.generator import SpecGenerator

router = APIRouter(tags=["specs"])


class GenerateSpecRequest(BaseModel):
    max_evidence_chunks: int = 60


def _generate_spec(ctx, paper_id: str, limit: int) -> dict:
    paper = ctx.db.get_paper(paper_id)
    generator = SpecGenerator(ctx.settings, ctx.llm)
    evidence = spec_evidence(ctx, paper_id, limit)
    spec = generator.generate(paper_id, paper.get("title"), evidence)
    ctx.db.insert_spec(spec)
    return {"spec_id": spec.id, "requirements": len(spec.requirements)}


@router.get("/papers/{paper_id}/specs")
def list_specs(paper_id: str, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)
    return {"specs": ctx.db.list_specs(paper_id)}


@router.post("/papers/{paper_id}/specs", status_code=202)
def generate_spec(
    paper_id: str, body: GenerateSpecRequest | None = None, ctx=Depends(get_ctx)
):
    require_paper(ctx, paper_id)
    limit = body.max_evidence_chunks if body else 60
    job_id = ctx.db.create_job(paper_id, "spec")
    run_job(ctx, job_id, _generate_spec, ctx, paper_id, limit)
    return {"job_id": job_id, "status": "queued"}


@router.get("/specs/{spec_id}")
def get_spec(spec_id: str, ctx=Depends(get_ctx)):
    spec = ctx.db.get_spec(spec_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="Spec not found")
    return spec


@router.patch("/specs/{spec_id}/requirements/{req_id}")
def update_requirement(
    spec_id: str,
    req_id: str,
    status: str,
    notes: str | None = None,
    ctx=Depends(get_ctx),
):
    if status not in ("open", "accepted", "rejected", "implemented"):
        raise HTTPException(status_code=400, detail="Invalid status")
    rows = ctx.db.query(
        "SELECT id FROM spec_requirements WHERE spec_id = ? AND req_id = ?",
        (spec_id, req_id),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Requirement not found")
    ctx.db.update_requirement_status(req_id, status, notes)
    return {"ok": True, "req_id": req_id, "status": status}
