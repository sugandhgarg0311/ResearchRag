"""Chunking contracts.

A chunker turns a structured ``Paper`` into retrieval units:

* **child chunks** — embedded & retrieved (``Chunk(kind="child")``)
* **parent chunks** — section-level context, expanded on demand
  (``Chunk(kind="parent")``)

The output of every strategy is the same schema (``researchrag.models.chunk``)
so downstream stages never need to know which strategy produced a chunk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from researchrag.models.chunk import Chunk
from researchrag.models.document import Paper


class BaseChunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, paper: Paper, paper_title: str | None = None) -> list[Chunk]:
        """Return parent + child chunks for the whole paper."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def children(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        return [c for c in chunks if c.kind == "child"]

    def parents(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        return [c for c in chunks if c.kind == "parent"]

    @staticmethod
    def child_by_id(chunks: Sequence[Chunk], chunk_id: str) -> Chunk | None:
        for c in chunks:
            if c.id == chunk_id:
                return c
        return None
