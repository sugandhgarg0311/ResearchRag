from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-base-en-v1.5"
    normalize_embeddings: bool = True
    batch_size: int = 32


embedding_config = EmbeddingConfig()