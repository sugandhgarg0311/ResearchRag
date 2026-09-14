"""Hybrid retrieval: dense + sparse → RRF → rerank → evidence candidates.

    Query
    ├── Dense retrieval  (semantic similarity, Qdrant cosine)
    └── Sparse retrieval (BM25: exact terms, identifiers, notation)
            │
            ▼
    Reciprocal Rank Fusion (rank-based; raw scores are never mixed)
            │
            ▼
    Top-M candidates
            │
            ▼
    Reranker (cross-encoder, or passthrough)
            │
            ▼
    Top-K evidence (with full per-stage observability)

Every stage's candidates, scores and timings are returned in the
``RetrievalResult`` so the UI's evidence/debug view and the experiment
runner can inspect exactly what happened.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from researchrag.config import Settings
from researchrag.models.chunk import Chunk
from researchrag.models.retrieval import (
    RetrievalConfig,
    RetrievalResult,
    RetrievedChunk,
    StageTimings,
)
from researchrag.retrieval.dense import DenseRetriever
from researchrag.retrieval.rerank import BaseReranker
from researchrag.retrieval.rrf import reciprocal_rank_fusion
from researchrag.retrieval.sparse import BM25Index

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    chunk: Chunk
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        dense: DenseRetriever | None,
        sparse_index: BM25Index | None,
        reranker: BaseReranker,
        chunk_loader,
    ):
        """chunk_loader(chunk_id) -> Chunk | None: fetches full chunk records
        from SQLite (Qdrant payloads are not the system of record for text).
        """
        self.settings = settings
        self.dense = dense
        self.sparse = sparse_index
        self.reranker = reranker
        self.chunk_loader = chunk_loader

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        paper_id: str | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        cfg = config or RetrievalConfig()
        timings = StageTimings()
        t_total = time.perf_counter()

        dense_top_n = cfg.dense_top_n or self.settings.dense_top_n
        sparse_top_n = cfg.sparse_top_n or self.settings.sparse_top_n
        rrf_k = cfg.rrf_k or self.settings.rrf_k
        top_m = cfg.reranker_top_m or self.settings.reranker_top_m
        top_k = cfg.final_top_k or self.settings.final_top_k
        use_dense = cfg.use_dense and self.dense is not None
        use_sparse = cfg.use_sparse and self.sparse is not None and self.sparse.doc_ids

        # ---------------- dense --------------------------------------
        candidates: dict[str, Candidate] = {}
        dense_ranking: list[str] = []
        if use_dense:
            t0 = time.perf_counter()
            hits = self.dense.search(query, paper_id, dense_top_n)
            timings.dense_ms = (time.perf_counter() - t0) * 1000
            for rank, h in enumerate(hits):
                cid = h["chunk_id"]
                chunk = self.chunk_loader(cid)
                if chunk is None:
                    continue
                cand = candidates.setdefault(cid, Candidate(chunk=chunk))
                cand.dense_rank = rank + 1
                cand.dense_score = h["score"]
                dense_ranking.append(cid)

        # ---------------- sparse -------------------------------------
        sparse_ranking: list[str] = []
        if use_sparse:
            t0 = time.perf_counter()
            hits = self.sparse.search(query, top_n=sparse_top_n)
            timings.sparse_ms = (time.perf_counter() - t0) * 1000
            for rank, h in enumerate(hits):
                chunk = self.chunk_loader(h.chunk_id)
                if chunk is None:
                    continue
                cand = candidates.setdefault(cid := h.chunk_id, Candidate(chunk=chunk))
                cand.sparse_rank = rank + 1
                cand.sparse_score = h.score
                sparse_ranking.append(h.chunk_id)

        # ---------------- fusion --------------------------------------
        t0 = time.perf_counter()
        rankings, weights = [], []
        if use_dense:
            rankings.append(dense_ranking)
            weights.append(cfg.rrf_dense_weight or self.settings.rrf_dense_weight)
        if use_sparse:
            rankings.append(sparse_ranking)
            weights.append(cfg.rrf_sparse_weight or self.settings.rrf_sparse_weight)

        fused: list[RetrievedChunk] = []
        if cfg.use_rrf and len(rankings) >= 2:
            for hit in reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights):
                cand = candidates.get(hit.doc_id)
                if cand is None:
                    continue
                fused.append(
                    RetrievedChunk(
                        chunk=cand.chunk,
                        dense_rank=cand.dense_rank,
                        dense_score=cand.dense_score,
                        sparse_rank=cand.sparse_rank,
                        sparse_score=cand.sparse_score,
                        rrf_score=hit.score,
                        rrf_rank=hit.rank,
                    )
                )
        else:
            # single-leg mode: keep that leg's order
            order = dense_ranking if use_dense else (sparse_ranking if use_sparse else [])
            for cid in order:
                cand = candidates.get(cid)
                if cand is None:
                    continue
                score = (
                    cand.dense_score
                    if use_dense
                    else (cand.sparse_score if use_sparse else 0.0)
                )
                fused.append(
                    RetrievedChunk(
                        chunk=cand.chunk,
                        dense_rank=cand.dense_rank,
                        dense_score=cand.dense_score,
                        sparse_rank=cand.sparse_rank,
                        sparse_score=cand.sparse_score,
                        rrf_score=score,
                        rrf_rank=len(fused) + 1,
                    )
                )
        timings.fusion_ms = (time.perf_counter() - t0) * 1000

        top_m_pool = fused[:top_m]

        # ---------------- rerank --------------------------------------
        final: list[RetrievedChunk] = top_m_pool
        if cfg.use_reranker and self.reranker.name != "none" and top_m_pool:
            t0 = time.perf_counter()
            results = self.reranker.rerank(
                query, [c.chunk for c in top_m_pool], top_m=len(top_m_pool)
            )
            timings.rerank_ms = (time.perf_counter() - t0) * 1000
            final = []
            for r in results:
                src = next(
                    (c for c in top_m_pool if c.chunk.id == r.chunk.id), None
                )
                if src is None:
                    continue
                final.append(
                    RetrievedChunk(
                        chunk=r.chunk,
                        dense_rank=src.dense_rank,
                        dense_score=src.dense_score,
                        sparse_rank=src.sparse_rank,
                        sparse_score=src.sparse_score,
                        rrf_score=src.rrf_score,
                        rrf_rank=src.rrf_rank,
                        rerank_score=r.score,
                        rerank_rank=r.rank,
                    )
                )
            final.sort(key=lambda c: -(c.rerank_score or 0.0))
        final = final[:top_k]

        timings.total_ms = (time.perf_counter() - t_total) * 1000
        log.info(
            "retrieval: dense=%d sparse=%d fused=%d reranked=%d final=%d "
            "(%.0fms total, reranker=%s)",
            len(dense_ranking),
            len(sparse_ranking),
            len(fused),
            len(top_m_pool),
            len(final),
            timings.total_ms,
            self.reranker.name,
        )
        return RetrievalResult(
            query=query,
            paper_id=paper_id,
            config=cfg,
            dense_candidates=len(dense_ranking),
            sparse_candidates=len(sparse_ranking),
            fused_candidates=len(fused),
            reranked_candidates=len(top_m_pool),
            final_count=len(final),
            evidence=final,
            stages=timings,
        )
