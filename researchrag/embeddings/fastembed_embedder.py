"""Dense embeddings via fastembed (ONNX Runtime backend).

Why fastembed / ONNX instead of sentence-transformers / PyTorch
---------------------------------------------------------------
* The selected model (bge-base-en-v1.5) is a standard transformer encoder.
  ONNX Runtime runs it on CPU with no torch dependency — ~10× smaller
  install and no CUDA driver required, while matching the reference
  implementation's vector space (same weights, same normalization).
* fastembed manages model download + caching and exposes a clean
  ``embed``/``query_embed`` API.
* Torch IS still a dependency of the optional ``[docling]`` extra (deep
  layout models). It is deliberately NOT required for the default pipeline:
  parsing (PyMuPDF), embeddings (ONNX), sparse (pure-Python BM25), and
  reranking (ONNX cross-encoder) all run torch-free.
"""

from __future__ import annotations

from collections.abc import Sequence

from fastembed import TextEmbedding

from researchrag.embeddings.base import BaseEmbedder


class FastEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 32,
        query_instruction: str = "",
        cache_dir: str | None = None,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._query_instruction = query_instruction
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
        )
        self._dim = int(self._model.dimension)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        clean = [t if t.strip() else " " for t in texts]
        vectors = self._model.embed(clean, batch_size=self._batch_size)
        return [list(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        if self._query_instruction:
            text = f"{self._query_instruction} {text}"
        vec = next(self._model.query_embed(text))
        return list(vec)
