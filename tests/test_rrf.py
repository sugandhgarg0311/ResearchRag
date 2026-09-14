"""Unit tests for Reciprocal Rank Fusion.

The original task explicitly requires RRF to be unit-tested for:
  * ties        — equal fused scores resolve deterministically
  * empty       — empty / absent rankings are handled
  * duplicates  — a doc repeated in a ranking counts once (at best position)
  * depth       — long rankings stay consistent and monotonic

RRF is rank-based: score(d) = Σ w_R / (k + rank_R(d)). It fuses ORDERINGS,
never raw scores, which is the core correctness property asserted here.
"""

from __future__ import annotations

import pytest

from researchrag.retrieval.rrf import RRFHit, fused_scores_to_dict, reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# empty / degenerate
# ---------------------------------------------------------------------------

def test_empty_rankings_list_is_empty():
    assert reciprocal_rank_fusion([]) == []


def test_all_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_empty_ranking():
    assert reciprocal_rank_fusion([[]]) == []


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a", "b"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a", "b"]], k=-5)


def test_weights_length_must_match():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


# ---------------------------------------------------------------------------
# basic correctness: the score formula
# ---------------------------------------------------------------------------

def test_single_ranking_preserves_order_and_scores():
    hits = reciprocal_rank_fusion([["a", "b", "c"]], k=10)
    assert [h.doc_id for h in hits] == ["a", "b", "c"]
    # score of rank r in a single weight-1 ranking = 1/(k+r)
    assert hits[0].score == pytest.approx(1 / 11)
    assert hits[1].score == pytest.approx(1 / 12)
    assert hits[2].score == pytest.approx(1 / 13)
    # 1-based rank
    assert [h.rank for h in hits] == [1, 2, 3]
    # ranks tuple has one entry (the single input ranking)
    assert hits[0].ranks == (1,)


def test_two_rankings_sum_the_inverse_rank_terms():
    # doc x is rank1 in R0 and rank3 in R1, k=60, w=1
    hits = reciprocal_rank_fusion([["x", "a", "b"], ["c", "d", "x"]], k=60)
    x = next(h for h in hits if h.doc_id == "x")
    assert x.score == pytest.approx(1 / (60 + 1) + 1 / (60 + 3))
    assert x.ranks == (1, 3)


def test_ranks_record_absence_as_zero():
    hits = reciprocal_rank_fusion([["a"], ["b"]], k=10)
    a = next(h for h in hits if h.doc_id == "a")
    b = next(h for h in hits if h.doc_id == "b")
    assert a.ranks == (1, 0)  # present in R0 at rank1, absent in R1
    assert b.ranks == (0, 1)


def test_fused_scores_to_dict():
    hits = reciprocal_rank_fusion([["a", "b"]])
    d = fused_scores_to_dict(hits)
    assert set(d) == {"a", "b"}
    assert all(isinstance(v, float) for v in d.values())


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------

def test_duplicate_within_a_ranking_counts_once_at_best_position():
    # "a" appears at rank1 and rank3 of the same ranking -> only rank1 counts.
    hits = reciprocal_rank_fusion([["a", "b", "a"]], k=10)
    a = next(h for h in hits if h.doc_id == "a")
    assert a.score == pytest.approx(1 / (10 + 1))
    assert a.ranks == (1,)


def test_duplicate_across_rankings_sums_both():
    # "a" rank1 in R0, rank1 in R1 -> 1/(k+1) twice.
    hits = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], k=10)
    a = next(h for h in hits if h.doc_id == "a")
    assert a.score == pytest.approx(2 / (10 + 1))
    assert a.ranks == (1, 1)
    assert a.rank == 1  # wins the fusion


def test_duplicate_best_position_wins_the_recorded_rank():
    # "a" rank2 in R0, rank1 in R1 -> best (lowest) single rank = 1
    hits = reciprocal_rank_fusion([["b", "a"], ["a", "c"]], k=10)
    a = next(h for h in hits if h.doc_id == "a")
    assert a.ranks == (2, 1)
    assert a.score == pytest.approx(1 / (10 + 2) + 1 / (10 + 1))


# ---------------------------------------------------------------------------
# ties: deterministic resolution
# ---------------------------------------------------------------------------

def test_tie_broken_by_best_single_rank():
    # "a" and "b" get identical fused scores, but "a" has a better single rank.
    # R0: a(1) b(2) ; R1: b(1) a(2)  -> both: 1/(k+1)+1/(k+2), tied.
    hits = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=10)
    assert hits[0].score == hits[1].score  # genuine tie
    # best single rank: a has rank1 (in R0), b has rank1 (in R1) -> still tied on
    # first_rank, so fall to first_seen: "a" was seen first (R0 scanned first).
    assert [h.doc_id for h in hits] == ["a", "b"]


def test_tie_is_stable_across_repeated_calls():
    r1 = reciprocal_rank_fusion([["z", "a", "m"], ["m", "a", "z"]], k=60)
    r2 = reciprocal_rank_fusion([["z", "a", "m"], ["m", "a", "z"]], k=60)
    assert [h.doc_id for h in r1] == [h.doc_id for h in r2]


def test_first_rank_breaks_score_tie():
    # Genuine score tie where best-single-rank differs. With k=1:
    #   "B": rank1 in R0, absent R1      -> 1/(1+1) = 0.5, best single rank = 1
    #   "A": rank3 in R0 and rank3 in R1 -> 1/4 + 1/4 = 0.5, best single rank = 3
    # Equal scores, but B has the better (lower) best single rank, so B fuses first.
    r0 = ["B", "x", "A"]
    r1 = ["y", "z", "A"]
    hits = reciprocal_rank_fusion([r0, r1], k=1)
    a = next(h for h in hits if h.doc_id == "A")
    b = next(h for h in hits if h.doc_id == "B")
    assert a.score == pytest.approx(b.score)  # real tie
    assert a.ranks == (3, 3)  # best single rank 3
    assert b.ranks == (1, 0)  # best single rank 1
    order = [h.doc_id for h in hits]
    assert order.index("B") < order.index("A")


def test_deterministic_order_for_full_swap():
    # Full swap between two rankings; every doc is tied. Order must be the
    # first-appearance order (R0 then R1), fully deterministic.
    hits = reciprocal_rank_fusion([["p", "q"], ["q", "p"]], k=60)
    assert [h.doc_id for h in hits] == ["p", "q"]


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def test_weights_scale_each_ranking():
    # With weight 2 on R0, "a" (rank1 R0) should clearly beat "b" (rank1 R1).
    hits = reciprocal_rank_fusion([["a"], ["b"]], k=10, weights=[2.0, 1.0])
    assert hits[0].doc_id == "a"
    assert hits[0].score == pytest.approx(2 / (10 + 1))
    assert hits[1].score == pytest.approx(1 / (10 + 1))


def test_zero_weight_ignores_a_ranking():
    # R1 has zero weight; only R0 contributes.
    hits = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=10, weights=[1.0, 0.0])
    # "a" only from R0 (rank1) -> 1/11. "b" only from R0 (rank2) -> 1/12.
    a = next(h for h in hits if h.doc_id == "a")
    b = next(h for h in hits if h.doc_id == "b")
    assert a.score == pytest.approx(1 / 11)
    assert b.score == pytest.approx(1 / 12)


# ---------------------------------------------------------------------------
# depth: long rankings
# ---------------------------------------------------------------------------

def test_depth_1000_docs_single_ranking_monotonic():
    n = 1000
    ids = [f"d{i:04d}" for i in range(n)]
    hits = reciprocal_rank_fusion([ids], k=60)
    assert len(hits) == n
    assert [h.doc_id for h in hits] == ids
    # scores strictly decreasing with rank
    scores = [h.score for h in hits]
    assert all(scores[i] > scores[i + 1] for i in range(n - 1))
    # rank field matches position
    assert hits[0].rank == 1
    assert hits[-1].rank == n


def test_depth_1000_docs_two_rankings_fused():
    n = 1000
    r0 = [f"d{i:04d}" for i in range(n)]
    r1 = list(reversed(r0))
    hits = reciprocal_rank_fusion([r0, r1], k=60)
    assert len(hits) == n
    # The docs that are near the top in BOTH rankings must surface.
    # d0000 is rank1 in r0 and rank n in r1; d0001 is rank2 in r0 and rank n-1 in r1.
    # d0000 and d0001 should be at or near the top of the fused list.
    top = [h.doc_id for h in hits[:2]]
    assert "d0000" in top
    # every doc appears exactly once
    assert len({h.doc_id for h in hits}) == n


def test_depth_consistency_repeated_calls():
    n = 300
    r0 = [f"x{i}" for i in range(n)]
    r1 = [f"x{i}" for i in range(0, n, 2)] + [f"x{i}" for i in range(1, n, 2)]
    a = [h.doc_id for h in reciprocal_rank_fusion([r0, r1], k=60)]
    b = [h.doc_id for h in reciprocal_rank_fusion([r0, r1], k=60)]
    assert a == b


def test_returns_rrfhit_instances():
    hits = reciprocal_rank_fusion([["a"]])
    assert all(isinstance(h, RRFHit) for h in hits)
