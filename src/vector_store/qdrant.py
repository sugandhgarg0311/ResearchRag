import uuid
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.chunking.base import Chunk
from src.vector_store.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        storage_path: str = "data/qdrant",
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Persistent local Qdrant storage
        self.client = QdrantClient(
            path=storage_path
        )

    def create_collection(self) -> None:

        # Don't recreate an existing collection
        if self.client.collection_exists(self.collection_name):
            print(
                f"Collection '{self.collection_name}' already exists."
            )
            return

        # Create collection with our embedding configuration
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Collection '{self.collection_name}' created."
        )

    def _to_qdrant_id(self, chunk_id: str) -> str:
        """
        Convert our canonical chunk ID into a deterministic
        UUID accepted by Qdrant.

        The original chunk ID is still stored in the payload.
        """

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk_id,
            )
        )

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:

        # -----------------------------------------------------
        # Validate chunk/vector alignment
        # -----------------------------------------------------

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunk/vector count mismatch: "
                f"{len(chunks)} chunks, "
                f"{len(vectors)} vectors"
            )

        points = []

        # -----------------------------------------------------
        # Create Qdrant points
        # -----------------------------------------------------

        for chunk, vector in zip(chunks, vectors):

            payload = {
                "chunk_id": chunk.id,
                "text": chunk.text,
                "document_id": chunk.document_id,
                **chunk.metadata,
            }

            point = PointStruct(
                id=self._to_qdrant_id(chunk.id),
                vector=vector,
                payload=payload,
            )

            points.append(point)

        # -----------------------------------------------------
        # Upsert into Qdrant
        #
        # New ID      → insert
        # Existing ID → replace/update
        # -----------------------------------------------------

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(
            f"Upserted {len(points)} points "
            f"into '{self.collection_name}'."
        )