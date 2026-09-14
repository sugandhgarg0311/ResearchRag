"""Unit tests for evaluation metrics (retrieval + answer-level, offline)."""

from __future__ import annotations

from researchrag.evaluation.metrics import (
    answer_relevance,
    citation_correctness,
    context_precision_at_k,
    context_recall,
    groundedness,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    retrieval_scores,
)

GOLD = {"a", "b", "c"}


# ---------------------------------------------------------------------------
# retrieval metrics
# ---------------------------------------------------------------------------

def test_recall_perfect_and_partial():
    assert recall_at_k(["a", "b", "c", "x"], GOLD, 4) == 1.0
    assert recall_at_k(["a", "x", "y", "z"], GOLD, 4) == 1 / 3
    assert recall_at_k([], GOLD, 5) == 0.0


def test_recall_empty_gold_is_zero():
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_precision():
    assert precision_at_k(["a", "b", "x", "y"], GOLD, 4) == 0.5
    assert precision_at_k(["a", "b"], GOLD, 2) == 1.0
    assert precision_at_k(["a"], GOLD, 0) == 0.0


def test_mrr_first_hit_and_miss():
    assert mrr(["x", "y", "a"], GOLD) == 1 / 3
    assert mrr(["a"], GOLD) == 1.0
    assert mrr(["x", "y"], GOLD) == 0.0


def test_ndcg_perfect_order():
    # all gold in top-3, ideal DCG == DCG
    assert ndcg_at_k(["a", "b", "c"], GOLD, 3) == 1.0


def test_ndcg_reduced_when_gold_is_later():
    early = ndcg_at_k(["a", "x", "y"], GOLD, 3)
    late = ndcg_at_k(["x", "y", "a"], GOLD, 3)
    assert early > late


def test_context_precision_binary():
    assert context_precision_at_k(["a", "x"], GOLD, 2) == 0.5
    assert context_precision_at_k([], GOLD, 5) == 0.0


def test_context_recall():
    assert context_recall(["a", "b"], GOLD, 2) == 2 / 3
    assert context_recall(["a", "b", "c"], GOLD, 3) == 1.0


def test_retrieval_scores_all_keys_present():
    s = retrieval_scores(["a", "x"], GOLD, k=2)
    for key in ("recall@k", "precision@k", "mrr", "ndcg@k",
                "context_precision@k", "context_recall@k"):
        assert key in s
    assert s["recall@k"] == round(1 / 3, 4)   # 1 of 3 gold in top-2
    assert s["precision@k"] == 0.5            # 1 of 2 retrieved is gold


# ---------------------------------------------------------------------------
# answer-level metrics
# ---------------------------------------------------------------------------

def test_citation_correctness():
    assert citation_correctness(["a", "b"], GOLD) == 1.0
    assert citation_correctness(["a", "x"], GOLD) == 0.5
    assert citation_correctness([], GOLD) == 0.0
    assert citation_correctness([], set()) == 1.0  # vacuously correct


def test_groundedness_supported_vs_fabricated():
    evidence = "The learning rate was 3e-5. We used the Adam optimizer. " \
               "Training ran for 1000 steps on GPUs."
    supported = "The learning rate was 3e-5 and we used the Adam optimizer."
    fabricated = "The model has 12 layers and was trained with SGD on TPU devices."
    assert groundedness(supported, evidence) > groundedness(fabricated, evidence)


def test_groundedness_empty_answer():
    assert groundedness("", "evidence text here") == 1.0


def test_answer_relevance_overlap():
    a = "The generator uses 6 encoder layers of BART."
    g = "The BART generator has 6 layers."
    assert answer_relevance(a, g) > 0.0
    assert answer_relevance("completely unrelated text here", g) < answer_relevance(a, g)


def test_answer_relevance_empty():
    assert answer_relevance("", "gold answer") == 0.0
    assert answer_relevance("answer text", "") == 0.0
