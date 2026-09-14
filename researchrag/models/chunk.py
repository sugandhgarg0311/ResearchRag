"""Chunk model: the unit of retrieval.

Hierarchy
---------
Document → Section → Parent context chunk → Child retrieval chunk.

* **Child chunks** are what get embedded and retrieved. They are built
  from paragraph/sentence boundaries *within one section*, and MAY SPAN
  MULTIPLE PHYSICAL PAGES (page boundaries are provenance, not semantic
  boundaries).
* **Parent chunks** hold the surrounding section context (capped) and are
  NOT embedded by default — they are retrieved *via* their children and
  used to expand evidence (parent-child / small-to-big retrieval).

Every chunk keeps exact provenance:

    document (paper) id, title, section path, page_start, page_end,
    block ids, parent id, source block types.

This is the contract used by embedding, indexing, retrieval, and the UI.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field


def make_chunk_id(paper_id: str, kind: str, ordinal: int, text: str) -> str:
    """Deterministic chunk id: stable across re-runs for identical content."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{paper_id}:{kind}:{ordinal:04d}:{digest}"


class Chunk(BaseModel):
    id: str
    paper_id: str
    paper_title: str | None = None

    kind: str = Field(description="'child' (retrieval unit) or 'parent' (context)")
    text: str
    token_count: int = 0

    # Hierarchy
    parent_id: str | None = Field(
        default=None, description="Parent context chunk this child belongs to"
    )
    section_path: list[str] = Field(default_factory=list)
    section_id: int | None = None

    # Physical provenance
    page_start: int
    page_end: int
    block_ids: list[str] = Field(default_factory=list)
    block_types: list[str] = Field(default_factory=list)

    # Content type: 'text' | 'table' | 'figure' | 'formula' | 'code' | 'section'
    chunk_type: str = "text"

    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def pages_display(self) -> str:
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"p.{self.page_start}–{self.page_end}"

    @property
    def section_display(self) -> str:
        return " › ".join(self.section_path) if self.section_path else "—"

    def provenance(self) -> dict[str, Any]:
        """Compact provenance record for citations / UI / logs."""
        return {
            "chunk_id": self.id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "section": self.section_display,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "pages": self.pages_display,
            "chunk_type": self.chunk_type,
            "block_ids": self.block_ids,
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_row(cls, row: dict) -> Chunk:
        """Build from a SQLite row (see storage.database._chunk_row)."""
        return cls(
            id=row["id"],
            paper_id=row["paper_id"],
            kind=row["kind"],
            text=row["text"],
            token_count=row.get("token_count", 0),
            parent_id=row.get("parent_id"),
            section_path=row.get("section_path", []),
            section_id=row.get("section_id"),
            page_start=row["page_start"],
            page_end=row["page_end"],
            block_ids=row.get("block_ids", []),
            chunk_type=row.get("chunk_type", "text"),
            metadata=row.get("metadata", {}),
        )
