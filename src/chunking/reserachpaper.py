"""
Section-aware chunker for ResearchRAG.

Strategy:

1. Preserve document reading order.
2. Respect section boundaries.
3. Split normal text using tiktoken.
4. Maximum 800 tokens per text chunk.
5. Use 100-token overlap.
6. Keep tables, figures, formulas and code as separate chunks.
7. Return the common Chunk object defined in base.py.
"""

from __future__ import annotations

import hashlib

import tiktoken

from src.chunking.base import BaseChunker, Chunk
from src.loaders.base import (
    DocumentBlock,
    LoadedDocument,
)


class SectionAwareChunker(BaseChunker):

    # These should not be arbitrarily split.
    ATOMIC_TYPES = {
        "table",
        "figure",
        "formula",
        "code",
    }

    def __init__(
        self,
        max_tokens: int = 800,
        overlap_tokens: int = 100,
        tokenizer_name: str = "cl100k_base",
    ):

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than 0"
            )

        if overlap_tokens < 0:
            raise ValueError(
                "overlap_tokens cannot be negative"
            )

        if overlap_tokens >= max_tokens:
            raise ValueError(
                "overlap_tokens must be smaller than max_tokens"
            )

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self.encoder = tiktoken.get_encoding(
            tokenizer_name
        )

    # ========================================================
    # PUBLIC METHOD
    # ========================================================

    def chunk(
        self,
        document: LoadedDocument,
    ) -> list[Chunk]:

        blocks = self._flatten_blocks(document)

        if not blocks:
            return []

        chunks: list[Chunk] = []

        text_blocks: list[DocumentBlock] = []

        current_section_path: list[str] = []

        for block in blocks:

            section_path = list(
                block.metadata.get(
                    "section_path",
                    [],
                )
            )

            # ------------------------------------------------
            # Section changed
            # ------------------------------------------------

            if (
                text_blocks
                and section_path != current_section_path
            ):

                chunks.extend(
                    self._create_text_chunks(
                        document=document,
                        blocks=text_blocks,
                        section_path=current_section_path,
                    )
                )

                text_blocks = []

            current_section_path = section_path

            # ------------------------------------------------
            # Atomic content
            # ------------------------------------------------

            if block.block_type in self.ATOMIC_TYPES:

                # Flush text before the atomic block.
                if text_blocks:

                    chunks.extend(
                        self._create_text_chunks(
                            document=document,
                            blocks=text_blocks,
                            section_path=current_section_path,
                        )
                    )

                    text_blocks = []

                chunks.extend(
                    self._create_atomic_chunk(
                        document=document,
                        block=block,
                        section_path=section_path,
                    )
                )

                continue

            # ------------------------------------------------
            # Normal text
            # ------------------------------------------------

            text_blocks.append(block)

        # ----------------------------------------------------
        # Remaining text
        # ----------------------------------------------------

        if text_blocks:

            chunks.extend(
                self._create_text_chunks(
                    document=document,
                    blocks=text_blocks,
                    section_path=current_section_path,
                )
            )

        # ----------------------------------------------------
        # Add final chunk IDs and indexes
        # ----------------------------------------------------

        for index, chunk in enumerate(chunks):

            chunk.metadata["chunk_index"] = index

            chunk.id = self._generate_chunk_id(
                document.document_id,
                index,
                chunk.text,
            )

        return chunks

    # ========================================================
    # FLATTEN DOCUMENT
    # ========================================================

    @staticmethod
    def _flatten_blocks(
        document: LoadedDocument,
    ) -> list[DocumentBlock]:

        blocks: list[DocumentBlock] = []

        for page in document.pages:
            blocks.extend(page.blocks)

        # Preserve original reading order.
        blocks.sort(
            key=lambda block: block.order
        )

        return blocks

    # ========================================================
    # NORMAL TEXT CHUNKS
    # ========================================================

    def _create_text_chunks(
        self,
        document: LoadedDocument,
        blocks: list[DocumentBlock],
        section_path: list[str],
    ) -> list[Chunk]:

        if not blocks:
            return []

        texts = [
            block.text.strip()
            for block in blocks
            if block.text.strip()
        ]

        if not texts:
            return []

        text = "\n\n".join(texts)

        tokens = self.encoder.encode(text)

        if not tokens:
            return []

        chunks: list[Chunk] = []

        start = 0

        while start < len(tokens):

            end = min(
                start + self.max_tokens,
                len(tokens),
            )

            chunk_tokens = tokens[start:end]

            chunk_text = self.encoder.decode(
                chunk_tokens
            ).strip()

            if chunk_text:

                pages = sorted(
                    {
                        block.page_number
                        for block in blocks
                    }
                )

                chunks.append(
                    Chunk(
                        id="",
                        document_id=document.document_id,
                        text=chunk_text,
                        metadata={
                            "document_name": (
                                document.document_name
                            ),
                            "source_path": (
                                document.source_path
                            ),
                            "mime_type": (
                                document.mime_type
                            ),
                            "page_number": pages[0],
                            "page_numbers": pages,
                            "section_path": list(
                                section_path
                            ),
                            "chunk_type": "text",
                            "token_count": len(
                                chunk_tokens
                            ),
                            "strategy": (
                                "section_aware"
                            ),
                        },
                    )
                )

            # Finished.
            if end >= len(tokens):
                break

            # 100-token overlap.
            start = (
                end - self.overlap_tokens
            )

        return chunks

    # ========================================================
    # TABLE / FIGURE / FORMULA / CODE
    # ========================================================

    def _create_atomic_chunk(
        self,
        document: LoadedDocument,
        block: DocumentBlock,
        section_path: list[str],
    ) -> list[Chunk]:

        text = block.text.strip()

        if not text:
            return []

        tokens = self.encoder.encode(text)

        # ----------------------------------------------------
        # Normal case:
        # keep the complete object together.
        # ----------------------------------------------------

        if len(tokens) <= self.max_tokens:

            return [
                Chunk(
                    id="",
                    document_id=document.document_id,
                    text=text,
                    metadata={
                        "document_name": (
                            document.document_name
                        ),
                        "source_path": (
                            document.source_path
                        ),
                        "mime_type": (
                            document.mime_type
                        ),
                        "page_number": (
                            block.page_number
                        ),
                        "page_numbers": [
                            block.page_number
                        ],
                        "section_path": list(
                            section_path
                        ),
                        "chunk_type": (
                            block.block_type
                        ),
                        "token_count": len(
                            tokens
                        ),
                        "strategy": (
                            "section_aware"
                        ),
                        "atomic": True,
                    },
                )
            ]

        # ----------------------------------------------------
        # Extremely large table/code/etc.
        #
        # Safety fallback so ingestion does not create an
        # oversized chunk.
        # ----------------------------------------------------

        chunks: list[Chunk] = []

        start = 0

        while start < len(tokens):

            end = min(
                start + self.max_tokens,
                len(tokens),
            )

            piece_tokens = tokens[start:end]

            piece = self.encoder.decode(
                piece_tokens
            ).strip()

            if piece:

                chunks.append(
                    Chunk(
                        id="",
                        document_id=document.document_id,
                        text=piece,
                        metadata={
                            "document_name": (
                                document.document_name
                            ),
                            "source_path": (
                                document.source_path
                            ),
                            "mime_type": (
                                document.mime_type
                            ),
                            "page_number": (
                                block.page_number
                            ),
                            "page_numbers": [
                                block.page_number
                            ],
                            "section_path": list(
                                section_path
                            ),
                            "chunk_type": (
                                block.block_type
                            ),
                            "token_count": len(
                                piece_tokens
                            ),
                            "strategy": (
                                "section_aware"
                            ),
                            "atomic": False,
                            "split_from_atomic": True,
                        },
                    )
                )

            start = end

        return chunks

    # ========================================================
    # CHUNK ID
    # ========================================================

    @staticmethod
    def _generate_chunk_id(
        document_id: str,
        index: int,
        text: str,
    ) -> str:

        raw = (
            f"{document_id}|"
            f"{index}|"
            f"{text}"
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()