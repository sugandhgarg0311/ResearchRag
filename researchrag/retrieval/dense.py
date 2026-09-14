"""Dense retrieval leg: query embedding → Qdrant cosine search."""

from __future__ import annotations

import logging
import time

from researchrag.embeddings.base import BaseEmbedder
from researchrag.storage.qdrant_store import QdrantStore

log = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, embedder: BaseEmbedder, store: QdrantStore):
        self.embedder = embedder
        self.store = store

    def search(
        self, query: str, paper_id: str | None, top_n: int
    ) -> list[dict]:
        t0 = time.perf_counter()
        vector = self.embedder.embed_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        hits = self.store.search(vector, paper_id=paper_id, limit=top_n)
        search_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "dense: embed %.0fms, search %.0fms, %d hits (paper=%s)",
            embed_ms, search_ms, len(hits), paper_id or "all",
        )
        return hits
