from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseEmbedder(ABC):
    """Symmetric/asymmetric document & query embedding."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
