from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Converts text into dense semantic vectors.

    Model:
        BAAI/bge-small-en-v1.5
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """
        Convert one piece of text into an embedding vector.
        """

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Convert multiple document chunks into embeddings.
        """

        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return vectors.tolist()

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Convert a user query into an embedding.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return vector.tolist()

    @property
    def dimension(self) -> int:
        """
        Return embedding vector dimension.
        """

        return self.model.get_embedding_dimension()