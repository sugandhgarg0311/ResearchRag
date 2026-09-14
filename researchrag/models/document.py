"""Structured document model for research papers.

A research paper is NOT a string. This module defines the canonical,
hierarchical representation that every stage of the pipeline uses:

    Paper
    └── Page
        └── Block          (title | heading | paragraph | list_item | table |
                            figure | formula | code | footnote | caption)
            └── Sentence   (lazy, computed when needed)

Design principles
-----------------
1. **Logical structure and physical provenance are separate concepts.**
   A block knows the page(s) it physically occupies; a chunk may span
   multiple pages while belonging to one logical section.
2. **Every unit carries provenance.** Every block id is globally unique
   (``{paper_id}:{page}:{order}``) and every chunk records the exact block
   ids it was built from, so any answer can be traced back to pages.
3. **Section hierarchy is explicit.** Each heading maintains a
   ``section_path`` (list of heading texts from root to leaf) and a stable
   ``section_id`` assigned in document order.
4. **Tables/figures/formulae/code are first-class blocks**, not text.

These models are pure data (no I/O). Persistence is handled by
``researchrag.storage``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    CODE = "code"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    TEXT = "text"  # unclassified text (kept for safety)


#: Block types that must never be split mid-block during chunking.
ATOMIC_BLOCK_TYPES = {
    BlockType.TABLE,
    BlockType.FIGURE,
    BlockType.FORMULA,
    BlockType.CODE,
    BlockType.CAPTION,
}


def make_block_id(paper_id: str, page: int, order: int) -> str:
    """Stable, sortable, human-readable block identifier."""
    return f"{paper_id}:{page:03d}:{order:04d}"


class Sentence(BaseModel):
    """A sentence within a block, with character offsets for provenance."""

    text: str
    start: int = 0
    end: int = 0
    index: int = 0


class Block(BaseModel):
    """One structural unit of the document with full provenance."""

    id: str
    paper_id: str
    block_type: BlockType
    text: str
    page: int
    page_end: int | None = Field(
        default=None, description="Last page occupied (for cross-page blocks)"
    )
    order: int = Field(description="Document reading order (global)")

    # Section hierarchy
    section_path: list[str] = Field(default_factory=list)
    section_id: int | None = None

    # Physical location on the page (PDF points, top-left origin)
    bbox: list[float] | None = None

    # Type-specific payload
    heading_level: int | None = None  # headings only
    caption: str | None = None  # figures / tables
    formula_latex: str | None = None  # when a LaTeX form is available
    image_path: str | None = None  # figures: path to extracted image
    description: str | None = None  # figures: optional VLM description
    table_rows: int | None = None  # tables
    table_cols: int | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def sentence_span(self, sentence_index: int) -> tuple[int, int] | None:
        return None  # computed lazily by chunking.sentences

    @property
    def span_pages(self) -> tuple[int, int]:
        return (self.page, self.page_end or self.page)


class Page(BaseModel):
    number: int
    width: float | None = None
    height: float | None = None
    blocks: list[Block] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)


class Paper(BaseModel):
    """Top-level structured document."""

    id: str
    filename: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    page_count: int
    mime_type: str = "application/pdf"
    file_sha256: str = ""
    size_bytes: int = 0
    source_path: str | None = None
    created_at: str = ""
    parser: str = "pymupdf"
    pages: list[Page] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    def all_blocks(self) -> list[Block]:
        blocks: list[Block] = []
        for page in self.pages:
            blocks.extend(page.blocks)
        return blocks

    def block_by_id(self, block_id: str) -> Block | None:
        for block in self.all_blocks():
            if block.id == block_id:
                return block
        return None

    @property
    def block_count(self) -> int:
        return sum(len(p.blocks) for p in self.pages)
