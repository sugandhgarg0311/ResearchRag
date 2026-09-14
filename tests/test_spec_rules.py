"""Unit tests for the rule-based spec extractor (offline fallback).

Contract under test: every extracted requirement is EXPLICIT, carries a
verbatim quote + page + chunk id, and nothing is invented. Texts with no
matches produce zero requirements (gaps are reported, not fabricated).
"""

from __future__ import annotations

from researchrag.models.retrieval import SourceKind
from researchrag.spec.extractor_rules import extract_requirements


def _evidence() -> list[tuple[str, str, list[str], int | None, str]]:
    return [
        (
            "c1",
            "We train with a learning rate of 3e-5 using AdamW for 1000 steps.",
            ["2 Methods", "2.4 Training"],
            4,
            "text",
        ),
        (
            "c2",
            "Training is distributed across 8, 32GB NVIDIA V100 GPUs.",
            ["C Training setup Details"],
            17,
            "text",
        ),
        (
            "c3",
            "We retrieve the top 1000 documents for each query.",
            ["D Details"],
            18,
            "text",
        ),
        (
            "c4",
            "The corpus contains 21M documents built from Wikipedia.",
            ["3 Experiments"],
            4,
            "table",
        ),
    ]


def test_every_requirement_is_explicit_with_provenance():
    spec = extract_requirements("p", "T", _evidence())
    assert spec.requirements, "no requirements extracted from known facts"
    for r in spec.requirements:
        assert r.source is SourceKind.EXPLICIT
        assert r.source_quote, "explicit requirement missing quote"
        assert r.source_page is not None
        assert r.source_chunk_id is not None


def test_known_facts_are_extracted():
    spec = extract_requirements("p", "T", _evidence())
    text = " ".join(r.requirement.lower() for r in spec.requirements)
    assert "3e-5" in text          # learning rate
    assert "adamw" in text          # optimizer
    assert "v100" in text           # hardware
    assert "n=1000" in text         # retrieval N ("Top-N documents ... N=1000")
    assert "21m documents" in text  # corpus size


def test_no_matches_produces_no_requirements():
    none = [("c9", "A paragraph with no implementation details at all.", [], 1, "text")]
    spec = extract_requirements("p", "T", none)
    assert spec.requirements == []


def test_duplicate_values_not_repeated():
    dup = [
        ("c1", "Learning rate of 1e-4 was used.", [], 1, "text"),
        ("c2", "The learning rate of 1e-4 is stated again.", [], 2, "text"),
    ]
    spec = extract_requirements("p", "T", dup)
    lrs = [r for r in spec.requirements if "learning rate" in r.requirement.lower()]
    assert len(lrs) == 1


def test_max_requirements_respected():
    # distinct learning-rate values so dedupe does not collapse them
    many = [
        (f"c{i}", f"We used a learning rate of 1e-{i % 9} on run {i}.", [], i + 1, "text")
        for i in range(30)
    ]
    spec = extract_requirements("p", "T", many, max_requirements=5)
    assert len(spec.requirements) <= 5


def test_spec_metadata():
    spec = extract_requirements("p", "My Paper", _evidence())
    assert spec.paper_id == "p"
    assert "My Paper" in spec.title
    assert spec.generated_by == "rule-extractor v1"
    assert spec.open_questions  # gaps are always reported
