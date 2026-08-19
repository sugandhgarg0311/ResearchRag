"""
Chunking Contract

Every chunking strategy (baseline, markdown-aware, csv-row-aware, table-aware,
semantic, ...) implements BaseChunker and returns List[Chunk]. This is the one
output shape all of them must agree on, so retrieval/embedding code downstream
never needs to know which strategy produced a given chunk.

METADATA KEYS every strategy should set:
  document_name  - document.document_name from the loader
  source_path    - document.source_path from the loader
  mime_type      - document.mime_type from the loader
  page_number    - which page the chunk came from
  chunk_index    - position of this chunk within its page (0-based)
  strategy       - name of the chunker that produced this chunk, e.g. "baseline"

Strategies may add extra metadata keys on top (e.g. char_start/char_end for
fixed-size windows, row_start/row_end for CSV, heading_path for markdown) but
must not omit the keys above.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List

from src.loaders.base import LoadedDocument


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: LoadedDocument) -> List[Chunk]:
        pass

    def chunk_documents(self, documents: List[LoadedDocument]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for document in documents:
            all_chunks.extend(self.chunk(document))
        return all_chunks

