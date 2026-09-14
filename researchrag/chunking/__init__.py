from researchrag.chunking.base import BaseChunker
from researchrag.chunking.fixed_size import FixedSizeChunker
from researchrag.chunking.section_aware import SectionAwareChunker

__all__ = ["BaseChunker", "FixedSizeChunker", "SectionAwareChunker"]


def make_chunker(strategy: str, **kwargs) -> BaseChunker:
    """kwargs use the section-aware names (max_tokens / parent_max_tokens).
    They are remapped for the fixed-size strategy, which uses its own names."""
    if strategy == "section_aware":
        return SectionAwareChunker(**kwargs)
    if strategy == "fixed_size":
        mapped = {
            "chunk_tokens": kwargs.get("max_tokens", 500),
            "overlap_tokens": kwargs.get("chunk_overlap_tokens", 80),
        }
        return FixedSizeChunker(**mapped)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
