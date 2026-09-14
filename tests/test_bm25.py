"""Unit tests for the BM25 sparse index (persistence + ranking)."""

from __future__ import annotations

import json

import pytest

from researchrag.retrieval.sparse import (
    BM25Index,
    delete_index,
    load_index,
    save_index,
    tokenize,
)


@pytest.fixture()
def corpus() -> list[tuple[str, str]]:
    return [
        ("c1", "retrieval augmented generation for knowledge intensive tasks"),
        ("c2", "the BART generator uses a denoising objective for pretraining"),
        ("c3", "we fine-tune the query encoder and the generator jointly"),
        ("c4", "exact match and F1 are reported on natural questions"),
    ]


@pytest.fixture()
def idx(corpus) -> BM25Index:
    return BM25Index().build(corpus)


def test_tokenize_keeps_identifiers_intact():
    assert "rag-token" in tokenize("RAG-Token models the latent")
    assert "top-k" in tokenize("we retrieve top-k documents")
    assert "state-of-the-art" in tokenize("state-of-the-art performance")


def test_best_matching_doc_ranks_first(idx):
    hits = idx.search("knowledge intensive tasks retrieval", top_n=3)
    assert hits[0].chunk_id == "c1"
    assert all(h.score > 0 for h in hits)


def test_empty_index_returns_empty():
    empty = BM25Index()
    assert empty.search("anything") == []


def test_no_match_returns_empty(idx):
    assert idx.search("zzz qqq xyz", top_n=5) == []


def test_top_n_limits_results(idx):
    hits = idx.search("the", top_n=2)
    assert len(hits) <= 2


def test_stats(idx):
    s = idx.stats()
    assert s["docs"] == 4
    assert s["vocab"] > 0
    assert s["avgdl"] > 0


def test_ranking_is_deterministic(idx):
    a = [h.chunk_id for h in idx.search("generator", top_n=4)]
    b = [h.chunk_id for h in idx.search("generator", top_n=4)]
    assert a == b


def test_persistence_roundtrip_preserves_ranking(idx, tmp_path):
    paper_id = "paper_test"
    save_index(idx, tmp_path, paper_id)
    loaded = load_index(tmp_path, paper_id)
    assert loaded is not None
    for q in ["retrieval augmented generation", "BART denoising", "fine-tune encoder"]:
        a = [h.chunk_id for h in idx.search(q, top_n=4)]
        b = [h.chunk_id for h in loaded.search(q, top_n=4)]
        assert a == b, f"ranking diverged for {q!r}"


def test_persistence_roundtrip_restores_int_posting_keys(idx, tmp_path):
    save_index(idx, tmp_path, "p")
    # simulate the JSON string-key path explicitly
    raw = json.loads((tmp_path / "index" / "bm25_p.json").read_text())
    assert any(isinstance(k, str) for posting in raw["inv"].values() for k in posting)
    loaded = BM25Index.from_dict(raw)
    # posting keys must be int again after from_dict
    assert all(
        isinstance(k, int) for posting in loaded._inv.values() for k in posting
    )


def test_load_missing_returns_none(tmp_path):
    assert load_index(tmp_path, "nope") is None


def test_delete_index(tmp_path):
    idx = BM25Index().build([("a", "hello world")])
    save_index(idx, tmp_path, "p")
    assert (tmp_path / "index" / "bm25_p.json").exists()
    delete_index(tmp_path, "p")
    assert not (tmp_path / "index" / "bm25_p.json").exists()
