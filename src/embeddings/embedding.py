from typing import Sequence

from sentence_transformers import SentenceTransformer

from src.config import embedding_config
from src.embeddings.base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):

    def __init__(self):
        self.model = SentenceTransformer(
            embedding_config.model_name
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:

        embeddings = self.model.encode(
            list(texts),
            batch_size=embedding_config.batch_size,
            normalize_embeddings=embedding_config.normalize_embeddings,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        embedding = self.model.encode(
            text,
            normalize_embeddings=embedding_config.normalize_embeddings,
        )

        return embedding.tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()