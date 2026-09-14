"""Reciprocal Rank Fusion (RRF).

Cormack, Clarke & Butt (2009), "Reciprocal Rank Fusion outperforms Condorcet
and individual Meta-learning rankings".

    score(d) = Σ over rankings R of:  w_R / (k + rank_R(d))

* Rank-based: dense cosine scores and BM25 scores are on incomparable
  scales; RRF fuses their *orderings*, which is exactly why we never add
  raw scores.
* ``k`` (default 60) damps the influence of top-rank differences; 60 is
  the value used in the original paper and in Elasticsearch/Qdrant.
* Per-ranking weights let experiments down-weight a weak leg (e.g. sparse
  on a small corpus) without changing the mechanism.

The function is pure and total: empty rankings, single rankings, ties,
duplicate documents, and mixed coverage are all defined and unit-tested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RRFHit:
    doc_id: str
    score: float
    rank: int  # 1-based rank in the fused list
    ranks: tuple[int, ...]  # rank in each input ranking (0 = absent)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RRFHit]:
    """Fuse one or more ranked id lists.

    Args:
        rankings: each element is an ordered list of document ids (best
            first). May be empty; duplicates within a ranking are counted
            once (at their first/best position).
        k: RRF constant (must be > 0).
        weights: per-ranking weights; default all ones.

    Returns:
        Fused hits sorted by score desc; ties broken by the best (lowest)
        single-ranking rank, then by first appearance — deterministic.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match rankings length")

    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}  # best rank across rankings (1-based)
    ranks_by_doc: dict[str, list[int]] = {}
    first_seen: dict[str, int] = {}

    for r_idx, ranking in enumerate(rankings):
        seen_in_ranking: set[str] = set()
        for position, doc_id in enumerate(ranking):
            if doc_id in seen_in_ranking:
                continue  # duplicates count once, at their best position
            seen_in_ranking.add(doc_id)
            rank = position + 1
            w = weights[r_idx]
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
            if doc_id not in first_rank or rank < first_rank[doc_id]:
                first_rank[doc_id] = rank
            ranks_by_doc.setdefault(doc_id, [0] * len(rankings))[r_idx] = rank
            if doc_id not in first_seen:
                first_seen[doc_id] = position

    ordered = sorted(
        scores,
        key=lambda d: (-scores[d], first_rank[d], first_seen[d]),
    )
    return [
        RRFHit(
            doc_id=d,
            score=scores[d],
            rank=i + 1,
            ranks=tuple(ranks_by_doc[d]),
        )
        for i, d in enumerate(ordered)
    ]


def fused_scores_to_dict(hits: Sequence[RRFHit]) -> Mapping[str, float]:
    return {h.doc_id: h.score for h in hits}
