from __future__ import annotations

import hashlib
from typing import List

import spacy
import tiktoken

from src.chunking.base import BaseChunker, Chunk
from src.loaders.base import (
    DocumentBlock,
    LoadedDocument,
)


class SectionAwareChunker(BaseChunker):
    """
    Section-aware, paragraph-aware, and sentence-aware chunker.

    Chunking hierarchy:

        Section
            |
            +-- Paragraph
            |      |
            |      +-- Sentence
            |
            +-- Paragraph
            |
            +-- Formula / Table / Figure / Code

    Rules
    -----
    1. Never mix different sections in the same chunk.
    2. Keep complete paragraphs together whenever possible.
    3. If adding a paragraph would exceed max_tokens,
       start a new chunk.
    4. If one paragraph itself exceeds max_tokens,
       split it at sentence boundaries.
    5. If one sentence itself exceeds max_tokens,
       fall back to a hard token split.
    6. Tables, figures, formulas, and code are treated as
       atomic blocks.
    7. Chunks do not cross page boundaries because the
       chunking contract requires a page_number.
    """

    ATOMIC_TYPES = {
        "table",
        "figure",
        "formula",
        "code",
    }

    def __init__(
        self,
        max_tokens: int = 800,
        tokenizer_name: str = "cl100k_base",
    ) -> None:

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than 0"
            )

        self.max_tokens = max_tokens

        self.encoder = tiktoken.get_encoding(
            tokenizer_name
        )

        # Lightweight sentence segmentation.
        # Does not require downloading a spaCy model.
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    # ================================================================
    # PUBLIC API
    # ================================================================

    def chunk(
        self,
        document: LoadedDocument,
    ) -> List[Chunk]:
        """
        Convert a LoadedDocument into chunks.

        Processing is page-by-page so that every chunk has an
        unambiguous page_number and chunk_index.
        """

        all_chunks: List[Chunk] = []

        for page in document.pages:

            # Keep document reading order.
            blocks = sorted(
                page.blocks,
                key=lambda block: block.order,
            )

            page_chunks = self._chunk_page(
                document=document,
                blocks=blocks,
            )

            # chunk_index is position WITHIN THIS PAGE.
            for chunk_index, chunk in enumerate(
                page_chunks
            ):
                chunk.metadata["chunk_index"] = (
                    chunk_index
                )

            all_chunks.extend(page_chunks)

        return all_chunks

    # ================================================================
    # PAGE
    # ================================================================

    def _chunk_page(
        self,
        document: LoadedDocument,
        blocks: List[DocumentBlock],
    ) -> List[Chunk]:
        """
        Chunk one page.

        We keep section boundaries intact while processing
        the blocks belonging to this page.
        """

        chunks: List[Chunk] = []

        current_section: tuple = ()
        current_text_blocks: List[DocumentBlock] = []

        for block in blocks:

            section_path = self._get_section_path(
                block
            )

            # --------------------------------------------------------
            # SECTION CHANGED
            # --------------------------------------------------------

            if section_path != current_section:

                # Flush text belonging to previous section.
                if current_text_blocks:

                    chunks.extend(
                        self._create_text_chunks(
                            document=document,
                            blocks=current_text_blocks,
                            section_path=list(
                                current_section
                            ),
                        )
                    )

                    current_text_blocks = []

                current_section = section_path

            # --------------------------------------------------------
            # ATOMIC BLOCK
            # --------------------------------------------------------

            if block.block_type in self.ATOMIC_TYPES:

                # Finish normal text before the atomic object.
                if current_text_blocks:

                    chunks.extend(
                        self._create_text_chunks(
                            document=document,
                            blocks=current_text_blocks,
                            section_path=list(
                                current_section
                            ),
                        )
                    )

                    current_text_blocks = []

                # Formula/table/figure/code gets its own chunk.
                chunks.extend(
                    self._create_atomic_chunks(
                        document=document,
                        block=block,
                        section_path=list(
                            current_section
                        ),
                    )
                )

                continue

            # --------------------------------------------------------
            # NORMAL TEXT
            # --------------------------------------------------------

            if block.text and block.text.strip():

                current_text_blocks.append(block)

        # Flush remaining text.
        if current_text_blocks:

            chunks.extend(
                self._create_text_chunks(
                    document=document,
                    blocks=current_text_blocks,
                    section_path=list(
                        current_section
                    ),
                )
            )

        return chunks

    # ================================================================
    # SECTION PATH
    # ================================================================

    def _get_section_path(
        self,
        block: DocumentBlock,
    ) -> tuple:
        """
        Read section hierarchy from block metadata.

        Expected example:

            metadata["section_path"] = [
                "3 Transformer Architecture",
                "3.1 Self-Attention"
            ]

        If no section information exists, use an empty path.
        """

        section_path = block.metadata.get(
            "section_path",
            [],
        )

        if section_path is None:
            return ()

        return tuple(section_path)

    # ================================================================
    # NORMAL TEXT
    # ================================================================

    def _create_text_chunks(
        self,
        document: LoadedDocument,
        blocks: List[DocumentBlock],
        section_path: List[str],
    ) -> List[Chunk]:
        """
        Create chunks from normal text blocks.

        Each DocumentBlock is treated as a paragraph.

        Example:

            Paragraph 1 -> 200 tokens
            Paragraph 2 -> 300 tokens
            Paragraph 3 -> 400 tokens

        max_tokens = 800

        Result:

            Chunk 1:
                Paragraph 1
                Paragraph 2

            Chunk 2:
                Paragraph 3
        """

        if not blocks:
            return []

        chunks: List[Chunk] = []

        current_blocks: List[DocumentBlock] = []
        current_tokens = 0

        for block in blocks:

            paragraph = block.text.strip()

            if not paragraph:
                continue

            paragraph_tokens = self.encoder.encode(
                paragraph
            )

            paragraph_token_count = len(
                paragraph_tokens
            )

            # --------------------------------------------------------
            # Paragraph fits inside max_tokens
            # --------------------------------------------------------

            if paragraph_token_count <= self.max_tokens:

                # It fits in current chunk.
                if (
                    current_tokens
                    + paragraph_token_count
                    <= self.max_tokens
                ):

                    current_blocks.append(block)

                    current_tokens += (
                        paragraph_token_count
                    )

                # It doesn't fit -> start new chunk.
                else:

                    chunks.append(
                        self._build_text_chunk(
                            document=document,
                            blocks=current_blocks,
                            text=self._join_paragraphs(
                                current_blocks
                            ),
                            section_path=section_path,
                        )
                    )

                    current_blocks = [block]
                    current_tokens = (
                        paragraph_token_count
                    )

                continue

            # --------------------------------------------------------
            # Paragraph itself is too large.
            # --------------------------------------------------------

            # Flush paragraphs already accumulated.
            if current_blocks:

                chunks.append(
                    self._build_text_chunk(
                        document=document,
                        blocks=current_blocks,
                        text=self._join_paragraphs(
                            current_blocks
                        ),
                        section_path=section_path,
                    )
                )

                current_blocks = []
                current_tokens = 0

            # Split this large paragraph by sentences.
            chunks.extend(
                self._split_large_paragraph(
                    document=document,
                    block=block,
                    section_path=section_path,
                )
            )

        # Flush remaining paragraphs.
        if current_blocks:

            chunks.append(
                self._build_text_chunk(
                    document=document,
                    blocks=current_blocks,
                    text=self._join_paragraphs(
                        current_blocks
                    ),
                    section_path=section_path,
                )
            )

        return chunks

    # ================================================================
    # LARGE PARAGRAPH
    # ================================================================

    def _split_large_paragraph(
        self,
        document: LoadedDocument,
        block: DocumentBlock,
        section_path: List[str],
    ) -> List[Chunk]:
        """
        Split an oversized paragraph at sentence boundaries.

        Example:

            Paragraph = 1500 tokens

            Sentence 1 = 200
            Sentence 2 = 300
            Sentence 3 = 250
            Sentence 4 = 350
            Sentence 5 = 400

        Result:

            Chunk 1 = sentences 1 + 2 + 3
            Chunk 2 = sentence 4
            Chunk 3 = sentence 5

        Assuming max_tokens = 800.
        """

        chunks: List[Chunk] = []

        doc = self.nlp(block.text.strip())

        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in doc.sents:

            sentence_text = sentence.text.strip()

            if not sentence_text:
                continue

            sentence_tokens = self.encoder.encode(
                sentence_text
            )

            sentence_token_count = len(
                sentence_tokens
            )

            # --------------------------------------------------------
            # Sentence itself fits.
            # --------------------------------------------------------

            if sentence_token_count <= self.max_tokens:

                if (
                    current_tokens
                    + sentence_token_count
                    <= self.max_tokens
                ):

                    current_sentences.append(
                        sentence_text
                    )

                    current_tokens += (
                        sentence_token_count
                    )

                else:

                    # Flush current sentence group.
                    chunks.append(
                        self._build_text_chunk(
                            document=document,
                            blocks=[block],
                            text=" ".join(
                                current_sentences
                            ),
                            section_path=section_path,
                        )
                    )

                    current_sentences = [
                        sentence_text
                    ]

                    current_tokens = (
                        sentence_token_count
                    )

            # --------------------------------------------------------
            # Sentence itself is too large.
            # --------------------------------------------------------

            else:

                # Flush sentences before it.
                if current_sentences:

                    chunks.append(
                        self._build_text_chunk(
                            document=document,
                            blocks=[block],
                            text=" ".join(
                                current_sentences
                            ),
                            section_path=section_path,
                        )
                    )

                    current_sentences = []
                    current_tokens = 0

                # Absolute last resort:
                # split the sentence by tokens.
                chunks.extend(
                    self._hard_token_split(
                        document=document,
                        block=block,
                        tokens=sentence_tokens,
                        section_path=section_path,
                    )
                )

        # Flush remaining sentences.
        if current_sentences:

            chunks.append(
                self._build_text_chunk(
                    document=document,
                    blocks=[block],
                    text=" ".join(
                        current_sentences
                    ),
                    section_path=section_path,
                )
            )

        return chunks

    # ================================================================
    # HARD TOKEN SPLIT
    # ================================================================

    def _hard_token_split(
        self,
        document: LoadedDocument,
        block: DocumentBlock,
        tokens: List[int],
        section_path: List[str],
    ) -> List[Chunk]:
        """
        Absolute fallback.

        This should only happen when ONE SENTENCE is itself
        larger than max_tokens.
        """

        chunks: List[Chunk] = []

        start = 0

        while start < len(tokens):

            end = min(
                start + self.max_tokens,
                len(tokens),
            )

            piece_tokens = tokens[start:end]

            text = self.encoder.decode(
                piece_tokens
            ).strip()

            if text:

                chunks.append(
                    self._build_text_chunk(
                        document=document,
                        blocks=[block],
                        text=text,
                        section_path=section_path,
                    )
                )

            start = end

        return chunks

    # ================================================================
    # ATOMIC BLOCKS
    # ================================================================

    def _create_atomic_chunks(
        self,
        document: LoadedDocument,
        block: DocumentBlock,
        section_path: List[str],
    ) -> List[Chunk]:
        """
        Create chunks for:

            formula
            table
            figure
            code

        Normally these remain completely intact.

        If one is larger than max_tokens, it is split as
        a last resort.
        """

        text = block.text.strip()

        if not text:
            return []

        tokens = self.encoder.encode(text)

        # ------------------------------------------------------------
        # Normal case: atomic object fits.
        # ------------------------------------------------------------

        if len(tokens) <= self.max_tokens:

            return [
                self._build_atomic_chunk(
                    document=document,
                    block=block,
                    text=text,
                    section_path=section_path,
                    atomic=True,
                )
            ]

        # ------------------------------------------------------------
        # Very large atomic object.
        # ------------------------------------------------------------

        chunks: List[Chunk] = []

        start = 0

        while start < len(tokens):

            end = min(
                start + self.max_tokens,
                len(tokens),
            )

            piece = self.encoder.decode(
                tokens[start:end]
            ).strip()

            if piece:

                chunks.append(
                    self._build_atomic_chunk(
                        document=document,
                        block=block,
                        text=piece,
                        section_path=section_path,
                        atomic=False,
                    )
                )

            start = end

        return chunks

    # ================================================================
    # BUILD TEXT CHUNK
    # ================================================================

    def _build_text_chunk(
        self,
        document: LoadedDocument,
        blocks: List[DocumentBlock],
        text: str,
        section_path: List[str],
    ) -> Chunk:

        text = text.strip()

        tokens = self.encoder.encode(text)

        page_number = blocks[0].page_number

        metadata = {
            # Required by BaseChunker contract.
            "document_name": document.document_name,
            "source_path": document.source_path,
            "mime_type": document.mime_type,
            "page_number": page_number,
            "chunk_index": -1,  # assigned later per page
            "strategy": "section_aware",

            # Additional useful metadata.
            "section_path": section_path,
            "chunk_type": "text",
            "token_count": len(tokens),
        }

        chunk_id = self._generate_chunk_id(
            document_id=document.document_id,
            page_number=page_number,
            text=text,
        )

        return Chunk(
            id=chunk_id,
            document_id=document.document_id,
            text=text,
            metadata=metadata,
        )

    # ================================================================
    # BUILD ATOMIC CHUNK
    # ================================================================

    def _build_atomic_chunk(
        self,
        document: LoadedDocument,
        block: DocumentBlock,
        text: str,
        section_path: List[str],
        atomic: bool,
    ) -> Chunk:

        tokens = self.encoder.encode(text)

        metadata = {
            # Required contract.
            "document_name": document.document_name,
            "source_path": document.source_path,
            "mime_type": document.mime_type,
            "page_number": block.page_number,
            "chunk_index": -1,  # assigned later per page
            "strategy": "section_aware",

            # Additional metadata.
            "section_path": section_path,
            "chunk_type": block.block_type,
            "token_count": len(tokens),
            "atomic": atomic,
        }

        chunk_id = self._generate_chunk_id(
            document_id=document.document_id,
            page_number=block.page_number,
            text=text,
        )

        return Chunk(
            id=chunk_id,
            document_id=document.document_id,
            text=text,
            metadata=metadata,
        )

    # ================================================================
    # HELPERS
    # ================================================================

    @staticmethod
    def _join_paragraphs(
        blocks: List[DocumentBlock],
    ) -> str:

        return "\n\n".join(
            block.text.strip()
            for block in blocks
            if block.text and block.text.strip()
        )

    @staticmethod
    def _generate_chunk_id(
        document_id: str,
        page_number: int,
        text: str,
    ) -> str:

        value = (
            f"{document_id}:"
            f"{page_number}:"
            f"{text}"
        )

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()