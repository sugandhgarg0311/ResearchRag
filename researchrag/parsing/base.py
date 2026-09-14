"""Parser interface.

A parser turns a raw file into the canonical structured ``Paper`` model
(see ``researchrag.models.document``). Parsers are responsible for:

* reading the file,
* reconstructing the section hierarchy,
* classifying blocks (heading / paragraph / table / figure / formula / code),
* preserving physical provenance (page, bbox),
* extracting figure images and table structures where available.

Parsers must NOT decide chunking — that is the chunker's job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from researchrag.models.document import Paper


class ParseError(Exception):
    """Raised when a document cannot be parsed into a usable structure."""


class BaseParser(ABC):
    name: str = "base"

    @abstractmethod
    def parse(self, path: Path, paper_id: str, artifacts_dir: Path) -> Paper:
        """Parse *path* and return a structured ``Paper``.

        Args:
            path: file to parse (PDF).
            paper_id: stable id assigned by the ingestion pipeline.
            artifacts_dir: directory where extracted assets (figure images)
                should be written.
        """
        raise NotImplementedError

    def supported(self) -> bool:
        """Whether the backend for this parser is installed/usable."""
        return True
