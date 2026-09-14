"""Ingestion pipeline: file → structured paper → chunks → indexes.

    PDF ─▶ parse (PyMuPDF / Docling)
        ─▶ normalize (structured Paper model)
        ─▶ chunk (section-aware hierarchical: parent + child)
        ─▶ embed (dense, ONNX)
        ─▶ index  (Qdrant dense  +  per-paper BM25 sparse)
        ─▶ persist (SQLite: paper, blocks, chunks; Qdrant: vectors)

Every step logs its duration and counts (the observability requirements).
The pipeline is idempotent per paper id (upserts), so re-ingestion after
config changes is safe.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from researchrag.chunking.base import BaseChunker
from researchrag.config import Settings
from researchrag.embeddings.base import BaseEmbedder
from researchrag.parsing.base import BaseParser
from researchrag.retrieval.sparse import BM25Index, save_index
from researchrag.storage.database import Database, new_id, now_iso
from researchrag.storage.qdrant_store import QdrantStore

log = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    paper_id: str
    title: str | None
    page_count: int
    block_count: int
    chunk_count: int
    parent_count: int
    dense_indexed: int
    sparse_indexed: int
    timings: dict[str, float] = field(default_factory=dict)
    parse_warnings: list[str] = field(default_factory=list)


ProgressFn = Callable[[float, str], None]


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        qdrant: QdrantStore,
        parser: BaseParser,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
    ):
        self.settings = settings
        self.db = db
        self.qdrant = qdrant
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder

    # ------------------------------------------------------------------
    def ingest_file(
        self,
        path: Path,
        progress: ProgressFn | None = None,
        paper_id: str | None = None,
    ) -> IngestionResult:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        def report(p: float, detail: str) -> None:
            log.info("ingest %s: %.0f%% %s", path.name, p * 100, detail)
            if progress:
                progress(p, detail)

        paper_id = paper_id or new_id("paper")
        self.db.upsert_paper(
            id=paper_id,
            filename=path.name,
            title=None,
            authors_json="[]",
            abstract=None,
            page_count=0,
            file_sha256="",
            size_bytes=path.stat().st_size,
            source_path=str(path),
            parser=self.parser.name,
            status="processing",
            error=None,
            created_at=now_iso(),
        )

        # ---------------- parse ---------------------------------------
        t0 = time.perf_counter()
        report(0.05, "parsing PDF")
        artifacts = self.settings.artifacts_root / paper_id
        artifacts.mkdir(parents=True, exist_ok=True)
        paper = self.parser.parse(path, paper_id, artifacts)
        timings["parse_s"] = round(time.perf_counter() - t0, 2)
        report(0.35, f"parsed {paper.page_count} pages, {paper.block_count} blocks")

        # ---------------- chunk ---------------------------------------
        t0 = time.perf_counter()
        report(0.40, "chunking")
        chunks = self.chunker.chunk(paper, paper.title)
        timings["chunk_s"] = round(time.perf_counter() - t0, 2)
        children = [c for c in chunks if c.kind == "child"]
        parents = [c for c in chunks if c.kind == "parent"]
        report(0.50, f"{len(children)} child + {len(parents)} parent chunks")

        # ---------------- embed ----------------------------------------
        t0 = time.perf_counter()
        report(0.55, "embedding chunks")
        vectors = self.embedder.embed_documents([c.text for c in children])
        timings["embed_s"] = round(time.perf_counter() - t0, 2)
        report(0.80, f"embedded {len(vectors)} chunks")

        # ---------------- index ----------------------------------------
        t0 = time.perf_counter()
        self.qdrant.upsert(children, vectors)
        timings["dense_index_s"] = round(time.perf_counter() - t0, 2)

        bm25 = BM25Index().build(
            [(c.id, c.text) for c in children]
        )
        save_index(bm25, self.settings.data_dir, paper_id)
        report(0.92, "indexed (dense + sparse)")

        # ---------------- persist ---------------------------------------
        t0 = time.perf_counter()
        all_blocks = paper.all_blocks()
        self.db.insert_blocks(paper_id, all_blocks)
        self.db.insert_chunks(paper_id, chunks)
        self.db.upsert_paper(
            id=paper_id,
            title=paper.title,
            authors_json=__import__("json").dumps(paper.authors),
            abstract=paper.abstract,
            page_count=paper.page_count,
            file_sha256=paper.file_sha256,
            parser=paper.parser,
            status="ready",
        )
        timings["persist_s"] = round(time.perf_counter() - t0, 2)

        timings["total_s"] = round(time.perf_counter() - t_start, 2)
        log.info("ingestion complete in %.1fs: %s", timings["total_s"], timings)
        return IngestionResult(
            paper_id=paper_id,
            title=paper.title,
            page_count=paper.page_count,
            block_count=paper.block_count,
            chunk_count=len(children),
            parent_count=len(parents),
            dense_indexed=len(vectors),
            sparse_indexed=bm25.stats()["docs"],
            timings=timings,
        )

    # ------------------------------------------------------------------
    def remove_paper(self, paper_id: str) -> None:
        self.qdrant.delete_paper(paper_id)
        from researchrag.retrieval.sparse import delete_index

        delete_index(self.settings.data_dir, paper_id)
        self.db.delete_paper(paper_id)
        artifacts = self.settings.artifacts_root / paper_id
        if artifacts.exists():
            import shutil

            shutil.rmtree(artifacts, ignore_errors=True)
        log.info("removed paper %s", paper_id)
