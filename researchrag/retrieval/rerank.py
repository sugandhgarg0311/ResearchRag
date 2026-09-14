"""Candidate reranking.

Pipeline position:  dense ∪ sparse → RRF → **top-M candidates** → reranker
→ top-K evidence. The reranker is only ever applied to a small candidate
set (default 30), never to the corpus.

Options (``RESEARCHRAG_RERANKER``)
----------------------------------
* ``local`` (default) — cross-encoder (Xenova/ms-marco-MiniLM-L-6-v2) via
  fastembed's ONNX backend. No torch. Weights (~25 MB) download from the
  model hub on first use; if unavailable the factory falls back to
  ``none`` with a warning (RRF order is preserved).
* ``cohere`` / ``jina`` — hosted reranking APIs (require API keys).
* ``none`` — passthrough (deterministic baseline for experiments).

Why a cross-encoder (and not late interaction / ColBERT)
--------------------------------------------------------
Cross-encoders give the best quality-per-latency for a 30-candidate set on
CPU; ColBERT-style late interaction buys marginal gains here at the cost of
large multi-vector indexes that this product does not need.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from researchrag.config import Settings
from researchrag.models.chunk import Chunk

log = logging.getLogger(__name__)


@dataclass
class RerankResult:
    chunk: Chunk
    score: float
    rank: int  # 1-based


class BaseReranker:
    name: str = "base"

    def rerank(
        self, query: str, candidates: Sequence[Chunk], top_m: int
    ) -> list[RerankResult]:
        raise NotImplementedError


class LocalCrossEncoderReranker(BaseReranker):
    name = "local"

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model_name = model_name
        self._enc = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, query: str, candidates: Sequence[Chunk], top_m: int
    ) -> list[RerankResult]:
        if not candidates:
            return []
        docs = [c.text for c in candidates]
        t0 = time.perf_counter()
        scores = list(self._enc.rerank(query, docs))
        log.debug("cross-encoder reranked %d in %.0fms", len(docs), (time.perf_counter() - t0) * 1000)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_m]
        return [
            RerankResult(chunk=candidates[i], score=float(scores[i]), rank=r + 1)
            for r, i in enumerate(order)
        ]


class NullReranker(BaseReranker):
    """Passthrough: keeps the incoming (RRF) order with neutral scores.

    Serves as the deterministic baseline in retrieval experiments and as
    the offline fallback when reranker weights cannot be downloaded.
    """

    name = "none"

    def rerank(
        self, query: str, candidates: Sequence[Chunk], top_m: int
    ) -> list[RerankResult]:
        return [
            RerankResult(chunk=c, score=0.0, rank=r + 1)
            for r, c in enumerate(candidates[:top_m])
        ]


class HostedReranker(BaseReranker):
    """Cohere or Jina hosted reranking (requires an API key)."""

    def __init__(self, provider: str, api_key: str, model: str | None = None):
        import httpx

        self._http = httpx.Client(timeout=60.0)
        self.name = provider
        if provider == "cohere":
            self._url = "https://api.cohere.com/v1/rerank"
            self._model = model or "rerank-multilingual-v3.0"
            self._headers = {"Authorization": f"Bearer {api_key}"}
        elif provider == "jina":
            self._url = "https://api.jina.ai/v1/rerank"
            self._model = model or "jina-reranker-v2-base-multilingual"
            self._headers = {"Authorization": f"Bearer {api_key}"}
        else:
            raise ValueError(f"Unknown hosted reranker: {provider}")

    def rerank(
        self, query: str, candidates: Sequence[Chunk], top_m: int
    ) -> list[RerankResult]:
        if not candidates:
            return []
        body = {
            "model": self._model,
            "query": query,
            "documents": [c.text for c in candidates],
            "top_n": top_m,
        }
        resp = self._http.post(self._url, json=body, headers=self._headers)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", []):
            idx = r.get("index")
            if idx is None or idx >= len(candidates):
                continue
            results.append(
                RerankResult(
                    chunk=candidates[idx],
                    score=float(r.get("relevance_score", r.get("score", 0.0))),
                    rank=0,
                )
            )
        results.sort(key=lambda x: -x.score)
        return [
            RerankResult(chunk=r.chunk, score=r.score, rank=i + 1)
            for i, r in enumerate(results[:top_m])
        ]


def make_reranker(settings: Settings) -> BaseReranker:
    kind = (settings.reranker or "local").lower()
    if kind == "none":
        return NullReranker()
    if kind in ("cohere", "jina"):
        key = settings.llm_api_key or ""
        if not key:
            log.warning(
                "Hosted reranker '%s' requires an API key; using 'none'.", kind
            )
            return NullReranker()
        return HostedReranker(kind, key)
    if kind == "local":
        try:
            reranker = LocalCrossEncoderReranker(settings.reranker_model)
            log.info("Local reranker ready: %s", settings.reranker_model)
            return reranker
        except Exception as e:
            log.warning(
                "Could not load local reranker %s (%s); falling back to 'none' "
                "(RRF order preserved). Connect to the network once to download "
                "the ~25 MB model.",
                settings.reranker_model,
                type(e).__name__,
            )
            return NullReranker()
    raise ValueError(f"Unknown reranker: {kind}")
