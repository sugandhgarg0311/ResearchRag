from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


@dataclass
class DocumentBlock:
    """
    One logical piece of document content.

    Examples:
        title
        heading
        text
        list_item
        table
        figure
        formula
        code
    """

    block_type: str
    text: str
    page_number: int
    order: int

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedPage:
    """
    Page-level provenance.

    Section hierarchy is stored on blocks because sections
    may span multiple pages.
    """

    blocks: list[DocumentBlock]
    page_number: int

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    """
    Standard representation returned by document loaders.
    """

    source_path: str
    document_name: str
    mime_type: str

    pages: list[LoadedPage]

    metadata: dict[str, Any] = field(default_factory=dict)

    # SHA-256 of original file.
    document_id: str = ""


class LoaderError(Exception):
    pass


class BaseLoader(ABC):

    @abstractmethod
    def load(self, path: Path) -> LoadedDocument:
        raise NotImplementedError