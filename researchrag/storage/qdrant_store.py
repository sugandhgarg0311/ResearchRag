"""Dense vector storage in Qdrant (embedded local mode).

Qdrant is used for what it is good at: vector similarity search over chunk
embeddings with structured payload filtering. It is NOT used as the
application database (that is SQLite).

One collection holds all child chunks from all papers; ``paper_id`` is a
payload field so queries filter to a single paper (multi-paper retrieval is
free).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from researchrag.models.chunk import Chunk

COLLECTION = "researchrag_chunks"


def qid(chunk_id: str) -> str:
    """Deterministic UUID for a canonical chunk id (Qdrant accepts UUIDs)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(self, storage_path: Path | str, collection: str = COLLECTION):
        self.collection = collection
        self.vector_size = 0
        self.client = QdrantClient(path=str(storage_path))

    def existing_dimension(self) -> int | None:
        try:
            info = self.client.get_collection(self.collection)
            cfg = info.config.params.vectors
            if hasattr(cfg, "size"):
                return int(cfg.size)
        except Exception:
            return None
        return None

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if absent; recreate if the dimension changed
        (a different embedding model was configured)."""
        current = self.existing_dimension()
        if current is None:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            self.vector_size = vector_size
        elif current != vector_size:
            import logging

            logging.getLogger(__name__).warning(
                "Vector dimension changed (%d → %d): recreating collection. "
                "Re-ingest all papers to repopulate dense vectors.",
                current,
                vector_size,
            )
            self.client.delete_collection(self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            self.vector_size = vector_size
        else:
            self.vector_size = vector_size

    # ------------------------------------------------------------------
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload: dict[str, Any] = {
                "chunk_id": chunk.id,
                "paper_id": chunk.paper_id,
                "paper_title": chunk.paper_title,
                "kind": chunk.kind,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "parent_id": chunk.parent_id,
                "section_path": chunk.section_path,
                "section_id": chunk.section_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "block_ids": chunk.block_ids,
            }
            points.append(
                PointStruct(
                    id=qid(chunk.id),
                    vector=list(vector),
                    payload=payload,
                )
            )
        # batch to keep memory flat on large corpora
        batch = 256
        for i in range(0, len(points), batch):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + batch],
                wait=False,
            )
        self.client.count(self.collection, exact=True)
        return len(points)

    def search(
        self,
        vector: Sequence[float],
        paper_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Dense search. Returns [{chunk_id, score, payload}, ...]."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if paper_id:
            query_filter = Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            )
        hits = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "chunk_id": h.payload["chunk_id"],
                "score": float(h.score),
                "payload": h.payload,
            }
            for h in hits.points
        ]

    def delete_paper(self, paper_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id", match=MatchValue(value=paper_id)
                        )
                    ]
                ),
            )
        except Exception:
            # nothing indexed for this paper yet
            pass

    def count(self, paper_id: str | None = None) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if paper_id:
            result = self.client.count(
                collection_name=self.collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id", match=MatchValue(value=paper_id)
                        )
                    ]
                ),
                exact=True,
            )
        else:
            result = self.client.count(self.collection, exact=True)
        return result.count

    def fetch(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        ids = [qid(c) for c in chunk_ids]
        if not ids:
            return []
        out = []
        batch = 256
        for i in range(0, len(ids), batch):
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=ids[i : i + batch],
                with_payload=True,
                with_vectors=False,
            )
            out.extend(p.payload for p in points)
        return out
