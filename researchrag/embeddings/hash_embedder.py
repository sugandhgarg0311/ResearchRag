"""Deterministic offline dense embedder (character n-gram hashing).

Purpose
-------
The primary dense embedder (bge-base-en-v1.5 via fastembed/ONNX) downloads
its weights from Hugging Face on first use. In environments where that
download is impossible (air-gapped machines, blocked egress), this embedder
keeps the full pipeline functional **out of the box**:

* 768-dimensional L2-normalized vectors,
* character n-grams (2..5) hashed into buckets with sublinear weights,
* deterministic — identical text always yields identical vectors.

Quality note: it captures lexical overlap and surface morphology, but not
the deep semantic similarity of a neural encoder. Treat it as a degraded
offline mode: sparse (BM25) + RRF + reranking still work, and the system
logs which embedder is active so results are never misleading.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from researchrag.embeddings.base import BaseEmbedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 768, ngram_range: tuple[int, int] = (2, 5)):
        self._dim = dim
        self._lo, self._hi = ngram_range

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"hashing-embedder/{self._dim}d"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        text = text.lower()
        grams: list[str] = []
        words = _TOKEN_RE.findall(text)
        for w in words:
            for n in range(self._lo, min(self._hi, len(w)) + 1):
                grams.extend(w[i : i + n] for i in range(len(w) - n + 1))
        # also raw character grams of the whole text for punctuation/notation
        compact = re.sub(r"\s+", "", text)
        for n in range(self._lo, min(self._hi, len(compact)) + 1):
            grams.extend(compact[i : i + n] for i in range(len(compact) - n + 1))
        if not grams:
            return vec
        for g in grams:
            digest = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        # sublinear weighting + L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t or " ") for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text or " ")
