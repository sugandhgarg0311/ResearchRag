"""Evaluation runner: dataset × retrieval config → metrics.

Retrieval evaluation (no LLM):
    For each question: retrieve top-K, score against gold chunks with
    Recall/Precision/MRR/NDCG/ContextPrecision/ContextRecall.

Answer evaluation (offline):
    citation correctness, groundedness, answer relevance against
    gold_answer, using the extractive answerer (no LLM needed).

LLM-judge evaluation (optional, when a key is available) plugs in here
later; the offline metrics are what CI can run anywhere.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from researchrag.answering.pipeline import AnsweringPipeline
from researchrag.evaluation.dataset import EvalDataset, resolve_gold_chunks
from researchrag.evaluation.metrics import (
    answer_relevance,
    citation_correctness,
    groundedness,
    retrieval_scores,
)
from researchrag.models.chunk import Chunk
from researchrag.models.retrieval import RetrievalConfig

log = logging.getLogger(__name__)


def evaluate_retrieval(
    ds: EvalDataset,
    chunks: list[Chunk],
    retrieve_fn: Callable[[str, RetrievalConfig], Any],
    k: int = 10,
) -> dict[str, Any]:
    gold_map = resolve_gold_chunks(ds, chunks)
    per_question: list[dict[str, Any]] = []
    totals: dict[str, list[float]] = {}
    for q in ds.questions:
        gold = gold_map.get(q.id, set())
        result = retrieve_fn(q.question, RetrievalConfig(final_top_k=k))
        retrieved = [rc.chunk.id for rc in result.evidence]
        scores = retrieval_scores(retrieved, gold, k=k)
        entry = {
            "id": q.id,
            "question": q.question,
            "class": q.question_class,
            "gold_count": len(gold),
            **scores,
        }
        per_question.append(entry)
        for mk, mv in scores.items():
            totals.setdefault(mk, []).append(mv)
    metrics = {
        mk: round(sum(vs) / len(vs), 4) if vs else 0.0 for mk, vs in totals.items()
    }
    by_class: dict[str, dict[str, float]] = {}
    for e in per_question:
        cls = e["class"]
        by_class.setdefault(cls, {"n": 0, "recall@k": 0.0, "ndcg@k": 0.0})
        by_class[cls]["n"] += 1
        by_class[cls]["recall@k"] += e["recall@k"]
        by_class[cls]["ndcg@k"] += e["ndcg@k"]
    for cls, agg in by_class.items():
        n = agg.pop("n")
        agg["recall@k"] = round(agg["recall@k"] / n, 4)
        agg["ndcg@k"] = round(agg["ndcg@k"] / n, 4)
    return {"metrics": metrics, "by_class": by_class, "per_question": per_question}


def evaluate_answers(
    ds: EvalDataset,
    answerer: AnsweringPipeline,
    paper_id: str,
    chunks: list[Chunk],
) -> dict[str, Any]:
    gold_map = resolve_gold_chunks(ds, chunks)
    per_question: list[dict[str, Any]] = []
    totals: dict[str, list[float]] = {}
    for q in ds.questions:
        t0 = time.perf_counter()
        ans = answerer.answer(q.question, paper_id)
        cited = [c.chunk_id for c in ans.citations]
        evidence_text = "\n".join(c.text for c in ans.evidence)
        gold = gold_map.get(q.id, set())
        entry = {
            "id": q.id,
            "question": q.question,
            "class": q.question_class,
            "mode": ans.mode,
            "citation_correctness": round(
                citation_correctness(cited, gold), 4
            ),
            "groundedness": round(groundedness(ans.answer, evidence_text), 4),
            "answer_relevance": (
                round(answer_relevance(ans.answer, q.gold_answer), 4)
                if q.gold_answer
                else None
            ),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        }
        per_question.append(entry)
        for mk in ("citation_correctness", "groundedness", "answer_relevance"):
            v = entry[mk]
            if v is not None:
                totals.setdefault(mk, []).append(v)
    metrics = {
        mk: round(sum(vs) / len(vs), 4) if vs else None for mk, vs in totals.items()
    }
    return {"metrics": metrics, "per_question": per_question}
