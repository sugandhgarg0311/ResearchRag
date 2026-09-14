"""Application context: the composition root.

Builds and caches every long-lived component from ``Settings``:

    Settings
    ├── Database (SQLite)
    ├── QdrantStore
    ├── Parser (pymupdf default / docling optional)
    ├── Chunker (section_aware default / fixed_size)
    ├── Embedder (fastembed ONNX, hashing fallback)
    ├── Reranker (local cross-encoder, hosted, or none)
    ├── LLM (openai_compatible / anthropic / offline)
    ├── IngestionPipeline
    └── per-paper HybridRetriever + AnsweringPipeline (built on demand)

Everything is constructed eagerly EXCEPT model weights, which load lazily
on first use — so the API boots fast even before the first paper.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

from researchrag.answering.pipeline import AnsweringPipeline
from researchrag.chunking import make_chunker
from researchrag.config import Settings, get_settings
from researchrag.embeddings.factory import make_embedder
from researchrag.evidence import EvidenceAssembler
from researchrag.ingestion.pipeline import IngestionPipeline
from researchrag.llm.factory import make_llm
from researchrag.parsing.base import BaseParser
from researchrag.parsing.pymupdf_parser import PymupdfParser
from researchrag.retrieval.hybrid import HybridRetriever
from researchrag.retrieval.rerank import make_reranker
from researchrag.storage.database import Database
from researchrag.storage.qdrant_store import QdrantStore

log = logging.getLogger(__name__)


def build_parser(settings: Settings) -> BaseParser:
    backend = (settings.parser_backend or "pymupdf").lower()
    if backend == "docling":
        from researchrag.parsing.docling_parser import DoclingParser

        parser = DoclingParser()
        if parser.supported():
            return parser
        log.warning(
            "Docling backend requested but not installed — using PyMuPDF. "
            "Install with: pip install 'researchrag[docling]'"
        )
    return PymupdfParser(figure_max_pixels=settings.figure_max_pixels)


class AppContext:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(settings.db_path)
        self.qdrant = QdrantStore(settings.qdrant_dir)
        self.parser = build_parser(settings)
        self._embedder, self._embedder_fell_back = make_embedder(settings)
        self.qdrant.ensure_collection(self._embedder.dimension)
        self.reranker = make_reranker(settings)
        self.llm = make_llm(settings)
        self.chunker = make_chunker(
            settings.chunk_strategy,
            max_tokens=settings.chunk_max_tokens,
            parent_max_tokens=settings.chunk_parent_max_tokens,
        )
        self.ingestion = IngestionPipeline(
            settings, self.db, self.qdrant, self.parser, self.chunker, self._embedder
        )
        self._retriever_cache: dict[str, HybridRetriever] = {}
        self._lock = threading.Lock()
        log.info(
            "AppContext ready (parser=%s, embedder=%s[fallback=%s], reranker=%s, llm=%s)",
            self.parser.name,
            self._embedder.model_name,
            self._embedder_fell_back,
            self.reranker.name,
            self.llm.name if self.llm else "offline",
        )

    # ------------------------------------------------------------------
    @property
    def embedder(self):
        return self._embedder

    # ------------------------------------------------------------------
    def _chunk_loader(self, chunk_id: str):
        from researchrag.models.chunk import Chunk

        row = self.db.get_chunk(chunk_id)
        return Chunk.from_row(row) if row else None

    def get_retriever(self, paper_id: str) -> HybridRetriever:
        from researchrag.retrieval.dense import DenseRetriever
        from researchrag.retrieval.sparse import load_index

        with self._lock:
            cached = self._retriever_cache.get(paper_id)
            if cached is not None:
                return cached
            sparse = load_index(self.settings.data_dir, paper_id)
            dense = DenseRetriever(self._embedder, self.qdrant)
            retriever = HybridRetriever(
                self.settings,
                dense=dense,
                sparse_index=sparse,
                reranker=self.reranker,
                chunk_loader=self._chunk_loader,
            )
            self._retriever_cache[paper_id] = retriever
            return retriever

    def invalidate_retriever(self, paper_id: str) -> None:
        with self._lock:
            self._retriever_cache.pop(paper_id, None)

    def get_answerer(self, paper_id: str, use_llm: bool = True) -> AnsweringPipeline:
        retriever = self.get_retriever(paper_id)
        assembler = EvidenceAssembler(
            max_tokens=self.settings.evidence_max_tokens,
            use_parent_context=self.settings.use_parent_context,
            parent_loader=self._chunk_loader,
        )
        return AnsweringPipeline(self.settings, retriever, self.llm if use_llm else None, assembler)


@lru_cache
def get_app_context(settings: Settings | None = None) -> AppContext:
    return AppContext(settings or get_settings())
