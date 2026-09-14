"""Retrieval & answer evaluation metrics (pure Python, no LLM required).

Retrieval metrics (against gold chunk id sets):
    Recall@K, Precision@K, MRR, NDCG@K, ContextPrecision@K, ContextRecall@K

Answer-level metrics (offline, deterministic):
    citation_correctness — cited chunks vs gold chunks (Jaccard/containment)
    groundedness         — answer sentences supported by cited evidence text
    answer_relevance     — term overlap between answer and gold answer
                           (a conservative offline stand-in for embedding
                           relevance; the LLM-judge variant needs a key)

These implement the standard definitions used by RAGAS-style evaluation
suits; the gold-set file format is defined in evaluation/dataset.py.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from researchrag.tokens import STOP_WORDS

# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = len(set(retrieved[:k]) & gold)
    return hit / len(gold)


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & gold) / k


def mrr(retrieved: Sequence[str], gold: set[str]) -> float:
    for i, doc in enumerate(retrieved):
        if doc in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Binary relevance (gold = 1)."""
    if not gold:
        return 0.0
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        if doc in gold:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def context_precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of top-k retrieved chunks that are gold (RAGAS context
    precision, binary-relevance form)."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & gold) / len(top)


def context_recall(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of gold chunks recovered in top-k."""
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def retrieval_scores(retrieved: Sequence[str], gold: set[str], k: int = 10) -> dict[str, float]:
    return {
        "recall@k": round(recall_at_k(retrieved, gold, k), 4),
        "precision@k": round(precision_at_k(retrieved, gold, k), 4),
        "mrr": round(mrr(retrieved, gold), 4),
        "ndcg@k": round(ndcg_at_k(retrieved, gold, k), 4),
        "context_precision@k": round(context_precision_at_k(retrieved, gold, k), 4),
        "context_recall@k": round(context_recall(retrieved, gold, k), 4),
    }


# ---------------------------------------------------------------------------
# answer metrics
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9][a-z0-9._\-]*")


def _terms(text: str) -> set[str]:
    """Content terms: lowercase tokens, length > 2, minus stopwords.

    Stopwords are excluded so that an unsupported sentence sharing only
    function words ("the", "was", "and") is NOT counted as grounded.
    """
    return {
        t
        for t in _WORD.findall(text.lower())
        if len(t) > 2 and t not in STOP_WORDS
    }


def citation_correctness(cited: Sequence[str], gold: set[str]) -> float:
    if not gold and not cited:
        return 1.0
    if not gold:
        return 0.0
    cited = set(cited)
    if not cited:
        return 0.0
    return len(cited & gold) / len(cited)


def groundedness(answer: str, evidence_text: str) -> float:
    """Fraction of answer sentences with >= 2 content terms present in the
    evidence. A conservative offline support check (catches fabrications
    with no basis in the retrieved text)."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 15]
    if not sentences:
        return 1.0
    ev = _terms(evidence_text)
    supported = 0
    for s in sentences:
        st = _terms(s)
        if st and len(st & ev) >= 2:
            supported += 1
    return supported / len(sentences)


def answer_relevance(answer: str, gold_answer: str) -> float:
    """Term-overlap relevance (offline stand-in for embedding similarity)."""
    a, g = _terms(answer), _terms(gold_answer)
    if not a or not g:
        return 0.0
    return len(a & g) / math.sqrt(len(a) * len(g))
