"""Section-aware hierarchical chunking (the default strategy).

Hierarchy and rules
-------------------
Paper
└── Section (from the parser's section stack)
    ├── parent chunk   (section context, capped at ``parent_max_tokens``)
    └── child chunks   (retrieval units, each ≤ ``max_tokens``)

1. Chunks are built from *logical* boundaries (sections / paragraphs /
   sentences). A chunk may SPAN MULTIPLE PHYSICAL PAGES whenever a section
   or paragraph continues across a page break — page boundaries are
   provenance metadata (``page_start``/``page_end``), never split points.
2. Paragraphs stay whole when they fit; oversized paragraphs split at
   sentence boundaries; as a last resort at word boundaries (never mid-word).
3. Tables, formulae, figures and code blocks are atomic: each gets its own
   child chunk (split only if it alone exceeds the token budget).
4. Captions are merged into their figure/table chunk so image/table
   retrieval works by caption text.
5. Every chunk records the exact ``block_ids`` it was built from, its
   ``section_path``, ``page_start``/``page_end`` and ``parent_id`` — full
   provenance for citation generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from researchrag.chunking.base import BaseChunker
from researchrag.chunking.sentences import normalize_ws, split_sentences
from researchrag.models.chunk import Chunk, make_chunk_id
from researchrag.models.document import ATOMIC_BLOCK_TYPES, Block, BlockType, Paper
from researchrag.tokens import TokenCounter, get_token_counter


def get_encoder(name: str | None = None) -> TokenCounter:
    """Kept for API compatibility; returns the active token counter."""
    return get_token_counter()


@dataclass
class _Section:
    path: list[str]
    section_id: int | None
    heading: Block | None
    text_blocks: list[Block] = field(default_factory=list)
    atomic_blocks: list[Block] = field(default_factory=list)


class SectionAwareChunker(BaseChunker):
    name = "section_aware"

    def __init__(
        self,
        max_tokens: int = 400,
        parent_max_tokens: int = 3000,
    ):
        if max_tokens <= 0 or parent_max_tokens <= 0:
            raise ValueError("token budgets must be positive")
        self.max_tokens = max_tokens
        self.parent_max_tokens = parent_max_tokens
        self.encoder = get_encoder()

    # ------------------------------------------------------------------
    def _tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def chunk(self, paper: Paper, paper_title: str | None = None) -> list[Chunk]:
        title = paper_title or paper.title
        sections = self._group_sections(paper)

        chunks: list[Chunk] = []
        child_ordinal = 0
        parent_ordinal = 0

        for sec in sections:
            # ----------------------------------------------------------
            # parent (section context) chunk
            # ----------------------------------------------------------
            parent_text_parts: list[str] = []
            if sec.heading is not None:
                parent_text_parts.append(sec.heading.text)
            for b in sec.text_blocks:
                parent_text_parts.append(b.text)
            for b in sec.atomic_blocks:
                parent_text_parts.append(b.text)
            parent_text = normalize_ws("\n".join(t for t in parent_text_parts if t.strip()))
            if not parent_text:
                continue
            parent_all_blocks = [sec.heading] if sec.heading is not None else []
            parent_all_blocks += sec.text_blocks + sec.atomic_blocks
            parent_page_start = min(b.page for b in parent_all_blocks)
            parent_page_end = max((b.page_end or b.page) for b in parent_all_blocks)

            if self._tokens(parent_text) > self.parent_max_tokens:
                # cap the parent context but keep the *beginning* (most
                # sections state their setup first)
                parent_text = self._truncate_tokens(parent_text, self.parent_max_tokens)
                parent_capped = True
            else:
                parent_capped = False

            parent_id = make_chunk_id(
                paper.id, "p", parent_ordinal, " > ".join(sec.path) or "front"
            )
            parent_ordinal += 1
            chunks.append(
                Chunk(
                    id=parent_id,
                    paper_id=paper.id,
                    paper_title=title,
                    kind="parent",
                    text=parent_text,
                    token_count=self._tokens(parent_text),
                    section_path=list(sec.path),
                    section_id=sec.section_id,
                    page_start=parent_page_start,
                    page_end=parent_page_end,
                    block_ids=[b.id for b in parent_all_blocks],
                    chunk_type="section",
                    metadata={"parent_capped": parent_capped},
                )
            )

            # ----------------------------------------------------------
            # child chunks: atomic blocks get their own chunk
            # ----------------------------------------------------------
            for b in sec.atomic_blocks:
                text = self._atomic_text(b)
                if not text:
                    continue
                for piece in self._fit(text, max(self.max_tokens, 64), atomic=True):
                    child_ordinal += 1
                    chunks.append(
                        self._child(
                            paper.id, title, child_ordinal, piece,
                            blocks=[b], parent_id=parent_id,
                            section_path=sec.path, section_id=sec.section_id,
                            chunk_type=b.block_type.value,
                            extra={"atomic": True},
                        )
                    )

            # ----------------------------------------------------------
            # child chunks: text blocks packed into ≤ max_tokens windows
            # ----------------------------------------------------------
            def flush_buffer(
                blocks: list[Block], ordinal: int
            ) -> tuple[list[Chunk], int]:
                nonlocal chunks
                emitted = self._emit_text_children(
                    paper.id, title, blocks, parent_id, sec, ordinal
                )
                chunks.extend(emitted)
                return emitted, ordinal + len(emitted)

            buffer: list[Block] = []
            buffer_tokens = 0
            for b in sec.text_blocks:
                text = normalize_ws(b.text)
                if not text:
                    continue
                b_tokens = self._tokens(text)
                if b_tokens <= self.max_tokens:
                    if buffer and buffer_tokens + b_tokens > self.max_tokens:
                        _, child_ordinal = flush_buffer(buffer, child_ordinal)
                        buffer, buffer_tokens = [], 0
                    buffer.append(b)
                    buffer_tokens += b_tokens
                else:
                    # paragraph bigger than the budget: flush, then split
                    if buffer:
                        _, child_ordinal = flush_buffer(buffer, child_ordinal)
                        buffer, buffer_tokens = [], 0
                    for piece in self._split_oversized(text):
                        child_ordinal += 1
                        chunks.append(
                            self._child(
                                paper.id, title, child_ordinal, piece,
                                blocks=[b], parent_id=parent_id,
                                section_path=sec.path, section_id=sec.section_id,
                                chunk_type="text",
                            )
                        )
            if buffer:
                _, child_ordinal = flush_buffer(buffer, child_ordinal)

        return chunks

    # ------------------------------------------------------------------
    # section grouping
    # ------------------------------------------------------------------
    def _group_sections(self, paper: Paper) -> list[_Section]:
        sections: list[_Section] = []
        current = _Section(path=[], section_id=None, heading=None)
        for b in paper.all_blocks():
            if b.block_type == BlockType.HEADING:
                sections.append(current)
                current = _Section(
                    path=list(b.section_path),
                    section_id=b.section_id,
                    heading=b,
                )
            elif b.block_type in ATOMIC_BLOCK_TYPES:
                current.atomic_blocks.append(b)
            elif b.block_type in (
                BlockType.PARAGRAPH,
                BlockType.LIST_ITEM,
                BlockType.TEXT,
                BlockType.FOOTNOTE,
            ):
                current.text_blocks.append(b)
            # TITLE / CAPTION blocks are metadata: the title is paper
            # metadata, and captions are already merged into their
            # figure/table chunk text (see _atomic_text).
            elif b.block_type in (BlockType.TITLE, BlockType.CAPTION):
                continue
        sections.append(current)
        return [s for s in sections if s.heading is not None or s.text_blocks or s.atomic_blocks]

    # ------------------------------------------------------------------
    # child emission
    # ------------------------------------------------------------------
    def _emit_text_children(
        self,
        paper_id: str,
        title: str | None,
        blocks: list[Block],
        parent_id: str,
        sec: _Section,
        start_ordinal: int,
    ) -> list[Chunk]:
        out: list[Chunk] = []
        if not blocks:
            return out
        text = normalize_ws("\n".join(b.text for b in blocks))
        ordinal = start_ordinal
        for piece in self._fit(text, self.max_tokens):
            ordinal += 1
            out.append(
                self._child(
                    paper_id, title, ordinal, piece,
                    blocks=blocks, parent_id=parent_id,
                    section_path=sec.path, section_id=sec.section_id,
                    chunk_type="text",
                )
            )
        return out

    def _child(
        self,
        paper_id: str,
        title: str | None,
        ordinal: int,
        text: str,
        blocks: list[Block],
        parent_id: str,
        section_path: list[str],
        section_id: int | None,
        chunk_type: str,
        extra: dict | None = None,
    ) -> Chunk:
        text = text.strip()
        return Chunk(
            id=make_chunk_id(paper_id, "c", ordinal, text),
            paper_id=paper_id,
            paper_title=title,
            kind="child",
            text=text,
            token_count=self._tokens(text),
            parent_id=parent_id,
            section_path=list(section_path),
            section_id=section_id,
            page_start=min(b.page for b in blocks),
            page_end=max((b.page_end or b.page) for b in blocks),
            block_ids=[b.id for b in blocks],
            block_types=sorted({b.block_type.value for b in blocks}),
            chunk_type=chunk_type,
            metadata={"strategy": self.name, **(extra or {})},
        )

    # ------------------------------------------------------------------
    # fitting / splitting helpers
    # ------------------------------------------------------------------
    def _atomic_text(self, b: Block) -> str:
        if b.block_type == BlockType.FIGURE:
            parts = []
            if b.caption:
                parts.append(b.caption)
            if b.description:
                parts.append(b.description)
            if not parts and b.text:
                parts.append(b.text)
            if not parts:
                return "[Figure]"
            return normalize_ws("\n".join(parts))
        if b.block_type == BlockType.TABLE:
            parts = []
            if b.caption:
                parts.append(b.caption)
            parts.append(b.text or "")
            return normalize_ws("\n".join(p for p in parts if p))
        return b.text or ""

    def _fit(self, text: str, budget: int, atomic: bool = False) -> list[str]:
        """Split *normalized* text into pieces ≤ budget tokens.

        Order of preference: whole → sentences → words.
        """
        if self._tokens(text) <= budget:
            return [text]
        if not atomic:
            # sentence-level packing
            pieces: list[str] = []
            buf: list[str] = []
            buf_tokens = 0
            for sent, _s, _e in split_sentences(text):
                t = self._tokens(sent)
                if t <= budget:
                    if buf and buf_tokens + t + 1 > budget:
                        pieces.append(" ".join(buf))
                        buf, buf_tokens = [], 0
                    buf.append(sent)
                    buf_tokens += t + 1
                else:
                    if buf:
                        pieces.append(" ".join(buf))
                        buf, buf_tokens = [], 0
                    pieces.extend(self._fit(sent, budget, atomic=True))
            if buf:
                pieces.append(" ".join(buf))
            return [p for p in pieces if p.strip()]
        # word-level (last resort; also used for atomic objects)
        words = text.split(" ")
        pieces = []
        buf: list[str] = []
        buf_tokens = 0
        for w in words:
            t = self._tokens(w) + 1
            if buf and buf_tokens + t > budget:
                pieces.append(" ".join(buf))
                buf, buf_tokens = [], 0
            buf.append(w)
            buf_tokens += t
        if buf:
            pieces.append(" ".join(buf))
        return [p for p in pieces if p.strip()]

    def _split_oversized(self, text: str) -> list[str]:
        return self._fit(text, self.max_tokens)

    def _truncate_tokens(self, text: str, budget: int) -> str:
        tokens = self.encoder.encode(text)
        if len(tokens) <= budget:
            return text
        cut = tokens[:budget]
        return self.encoder.decode(cut).strip()
