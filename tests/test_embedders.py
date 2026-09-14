"""Unit tests for the offline hashing embedder (deterministic, normalized)."""

from __future__ import annotations

import math

from researchrag.embeddings.hash_embedder import HashingEmbedder


def _l2(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def test_dimension():
    e = HashingEmbedder(dim=768)
    assert e.dimension == 768
    v = e.embed_query("hello world")
    assert len(v) == 768


def test_deterministic():
    e = HashingEmbedder()
    a = e.embed_documents(["the quick brown fox"])[0]
    b = e.embed_query("the quick brown fox")
    assert a == b


def test_l2_normalized():
    e = HashingEmbedder()
    for text in ["retrieval augmented generation", "ab", "hello world 123"]:
        v = e.embed_query(text)
        assert abs(_l2(v) - 1.0) < 1e-6, f"not normalized: {text!r}"


def test_no_grams_is_zero_vector():
    e = HashingEmbedder()
    # a single character produces no n-grams (n >= 2) -> zero vector
    v = e._vector("a")
    assert all(x == 0.0 for x in v)


def test_similar_texts_closer_than_dissimilar():
    e = HashingEmbedder()
    q = e.embed_query("retrieval augmented generation")
    near = e.embed_documents(["retrieval augmented generation models"])[0]
    far = e.embed_documents(["the stock market crashed on tuesday"])[0]
    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    assert dot(q, near) > dot(q, far)


def test_embed_documents_batch():
    e = HashingEmbedder(dim=64)
    vecs = e.embed_documents(["one", "two", "three"])
    assert len(vecs) == 3
    assert all(len(v) == 64 for v in vecs)


def test_model_name_reports_dimension():
    assert "64d" in HashingEmbedder(dim=64).model_name
