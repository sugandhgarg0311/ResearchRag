"""Fixed-size sliding-window chunking (baseline for retrieval experiments).

Flattens the document's content blocks in reading order and cuts windows of
``chunk_tokens`` with ``overlap_tokens`` overlap. It exists so experiments can
measure *why* section-aware chunking is better: it has no notion of sections,
keeps no parent context, and splits mid-sentence freely.
"""

from __future__ import annotations

from dataclasses import dataclass

from researchrag.chunking.base import BaseChunker
from researchrag.chunking.section_aware import get_encoder
from researchrag.models.chunk import Chunk, make_chunk_id
from researchrag.models.document import Block, BlockType, Paper


@dataclass
class _Piece:
    text: str
    tokens: int
    block: Block


class FixedSizeChunker(BaseChunker):
    name = "fixed_size"

    def __init__(self, chunk_tokens: int = 500, overlap_tokens: int = 80):
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be in [0, chunk_tokens)")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = get_encoder()

    def chunk(self, paper: Paper, paper_title: str | None = None) -> list[Chunk]:
        title = paper_title or paper.title
        pieces: list[_Piece] = []
        for b in paper.all_blocks():
            if b.block_type not in (
                BlockType.PARAGRAPH,
                BlockType.LIST_ITEM,
                BlockType.TEXT,
                BlockType.TABLE,
                BlockType.FORMULA,
                BlockType.FOOTNOTE,
            ):
                continue
            t = b.text.strip()
            if t:
                pieces.append(_Piece(t, len(self.encoder.encode(t)), b))

        chunks: list[Chunk] = []
        cur: list[_Piece] = []
        cur_tokens = 0
        ordinal = 0

        def flush() -> None:
            nonlocal cur, cur_tokens, ordinal
            if not cur:
                return
            text = " ".join(p.text for p in cur)
            blocks = [p.block for p in cur]
            ordinal += 1
            chunks.append(
                Chunk(
                    id=make_chunk_id(paper.id, "c", ordinal, text),
                    paper_id=paper.id,
                    paper_title=title,
                    kind="child",
                    text=text,
                    token_count=len(self.encoder.encode(text)),
                    page_start=min(b.page for b in blocks),
                    page_end=max((b.page_end or b.page) for b in blocks),
                    block_ids=[b.id for b in blocks],
                    chunk_type="text",
                    metadata={"strategy": self.name},
                )
            )
            cur, cur_tokens = [], 0

        for p in pieces:
            if cur and cur_tokens + p.tokens + 1 > self.chunk_tokens:
                flush()
                # carry overlap from the tail of the chunk just flushed
                if self.overlap_tokens > 0 and chunks:
                    tail_ids = self.encoder.encode(chunks[-1].text)[
                        -self.overlap_tokens :
                    ]
                    tail_text = self.encoder.decode(tail_ids).strip()
                    if tail_text:
                        cur.append(_Piece(tail_text, len(tail_ids), p.block))
                        cur_tokens = len(tail_ids)
            cur.append(p)
            cur_tokens += p.tokens + 1
        flush()
        return chunks
