from typing import List

from src.loaders.base import LoadedDocument
from src.chunking.base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: LoadedDocument) -> List[Chunk]:

        # Combine all page text into one document-level text
        text = "\n".join(
            page.text for page in document.pages
        )

        if not text.strip():
            return []

        chunks = []

        step = self.chunk_size - self.overlap

        start = 0

        while start < len(text):

            end = min(
                start + self.chunk_size,
                len(text)
            )

            chunk_text = text[start:end]

            chunks.append(
                Chunk(
                    id=(
                        f"{document.document_id}"
                        f"_chunk_{len(chunks):04d}"
                    ),

                    document_id=document.document_id,

                    text=chunk_text,

                    metadata={
                        "document_name": document.document_name,
                        "source_path": document.source_path,
                        "mime_type": document.mime_type,

                        "chunk_index": len(chunks),

                        "start_char": start,
                        "end_char": end,

                        "chunking_strategy": "fixed_size",

                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                    },
                )
            )

            start += step

        # Add total_chunks after all chunks are created
        total_chunks = len(chunks)

        for chunk in chunks:
            chunk.metadata["total_chunks"] = total_chunks

        return chunks