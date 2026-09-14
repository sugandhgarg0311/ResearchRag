"""Retrieval architecture experiments.

Compares configurations on a fixed eval dataset for one paper and stores
the results (SQLite `experiments`), so the project can *demonstrate* why
the chosen architecture wins:

    chunking:        section_aware  vs  fixed_size
    retrieval:       dense_only  vs  sparse_only  vs  hybrid
    fusion:          with/without RRF (concatenated passthrough when off)
    reranking:       on (when a real reranker is loaded)  vs  off
    sweeps:          top-K in {5, 10}

Each experiment row is one configuration; results are mean retrieval
metrics over the eval dataset. All variants run in-memory against the
paper's blocks, so a full experiment takes seconds.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from researchrag.config import Settings
from researchrag.evaluation.dataset import EvalDataset, resolve_gold_chunks
from researchrag.evaluation.metrics import retrieval_scores
from researchrag.models.retrieval import RetrievalConfig
from researchrag.retrieval.rrf import reciprocal_rank_fusion
from researchrag.retrieval.sparse import BM25Index
from researchrag.storage.database import Database, now_iso

log = logging.getLogger(__name__)

K_SWEPT = (5, 10)
CANDIDATE_POOL = 30  # candidates considered per config (before top-K cut)

BASE_CONFIGS = [
    ("dense_only", RetrievalConfig(use_dense=True, use_sparse=False)),
    ("sparse_only", RetrievalConfig(use_dense=False, use_sparse=True, use_rrf=False)),
    ("hybrid_rrf", RetrievalConfig(use_dense=True, use_sparse=True, use_rrf=True)),
    ("hybrid_concat", RetrievalConfig(use_dense=True, use_sparse=True, use_rrf=False)),
]


def run_retrieval_experiment(
    settings: Settings,
    db: Database,
    name: str,
    paper_id: str,
    dataset: EvalDataset,
    chunks_by_strategy: dict[str, list],
    bm25_by_strategy: dict[str, BM25Index],
    dense_search_fn,
    reranker,
) -> dict[str, Any]:
    """chunks_by_strategy: strategy -> list[Chunk] (children)
    dense_search_fn(strategy, query, k) -> list[(chunk_id, score)]
    """
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []

    for strategy, chunks in chunks_by_strategy.items():
        gold_map = resolve_gold_chunks(dataset, chunks)
        for k in K_SWEPT:
            for cfg_name, cfg in BASE_CONFIGS:
                # rerank only changes ordering when a real reranker is loaded
                agg: dict[str, list[float]] = {}
                for q in dataset.questions:
                    retrieved = _retrieve_with_config(
                        query=q.question,
                        strategy=strategy,
                        cfg=cfg,
                        chunks=chunks,
                        bm25=bm25_by_strategy[strategy],
                        dense_search_fn=dense_search_fn,
                        reranker=reranker,
                        top_n=settings.dense_top_n,
                        top_k=k,
                    )
                    scores = retrieval_scores(retrieved, gold_map.get(q.id, set()), k=k)
                    for mk, mv in scores.items():
                        agg.setdefault(mk, []).append(mv)
                results.append(
                    {
                        "strategy": strategy,
                        "retrieval": cfg_name,
                        "rerank": "on" if reranker is not None and reranker.name != "none" else "off",
                        "top_k": k,
                        **{mk: round(sum(vs) / len(vs), 4) for mk, vs in agg.items()},
                    }
                )

    duration = int((time.perf_counter() - t0) * 1000)
    exp = {
        "id": f"exp_{uuid.uuid4().hex[:10]}",
        "name": name,
        "paper_id": paper_id,
        "kind": "retrieval",
        "config": {
            "strategies": list(chunks_by_strategy),
            "k_swept": list(K_SWEPT),
            "n_questions": len(dataset.questions),
        },
        "results": {"rows": results},
        "created_at": now_iso(),
        "duration_ms": duration,
    }
    db.insert_experiment(exp)
    log.info("experiment %s: %d rows in %dms", name, len(results), duration)
    return exp


def _retrieve_with_config(
    query: str,
    strategy: str,
    cfg: RetrievalConfig,
    chunks,
    bm25: BM25Index,
    dense_search_fn,
    reranker,
    top_n: int,
    top_k: int,
) -> list[str]:
    chunk_by_id = {c.id: c for c in chunks}

    dense_ranking: list[str] = []
    sparse_ranking: list[str] = []
    if cfg.use_dense:
        dense_ranking = [h[0] for h in dense_search_fn(strategy, query, top_n)]
    if cfg.use_sparse:
        sparse_ranking = [h.chunk_id for h in bm25.search(query, top_n=top_n)]

    # RRF is rank-based: only list order matters, never scores
    if cfg.use_rrf and dense_ranking and sparse_ranking:
        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
        ordered = [h.doc_id for h in fused]
    else:
        seen = set(dense_ranking)
        ordered = list(dense_ranking) + [c for c in sparse_ranking if c not in seen]

    candidates = [chunk_by_id[c] for c in ordered if c in chunk_by_id][:CANDIDATE_POOL]
    if reranker is not None and reranker.name != "none" and len(candidates) > 1:
        rr = reranker.rerank(query, candidates, top_m=len(candidates))
        ordered = [r.chunk.id for r in rr]
    return ordered[:top_k]
