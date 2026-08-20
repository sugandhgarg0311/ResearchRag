from abc import ABC, abstractmethod
from typing import Sequence

from src.chunking.base import Chunk


class BaseVectorStore(ABC):

    @abstractmethod
    def create_collection(self) -> None:
        pass

    @abstractmethod
    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        pass