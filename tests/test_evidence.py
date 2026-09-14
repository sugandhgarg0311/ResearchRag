"""Unit tests for evidence assembly (parent context, budget, citations)."""

from __future__ import annotations

from researchrag.evidence import EvidenceAssembler
from researchrag.models.chunk import Chunk
from researchrag.models.retrieval import RetrievedChunk
from researchrag.tokens import get_token_counter


def _chunk(paper_id: str, cid: str, text: str, parent_id=None,
           page_start=1, page_end=1, section=None) -> Chunk:
    return Chunk(
        id=cid,
        paper_id=paper_id,
        paper_title="T",
        kind="child",
        text=text,
        token_count=10,
        parent_id=parent_id,
        section_path=section or ["S"],
        page_start=page_start,
        page_end=page_end,
        block_ids=[f"b{cid}"],
    )


def _rc(chunk: Chunk, rrf_score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, rrf_score=rrf_score)


def test_parent_header_emitted_once_per_section():
    parent = _chunk("p", "parent-1", "section text", section=["2 Methods"])
    c1 = _chunk("p", "c1", "first child", parent_id="parent-1", section=["2 Methods"])
    c2 = _chunk("p", "c2", "second child", parent_id="parent-1", section=["2 Methods"])
    assembler = EvidenceAssembler(
        max_tokens=4000, use_parent_context=True, parent_loader=lambda pid: parent
    )
    pkg = assembler.assemble([_rc(c1), _rc(c2)])
    assert "[context: 2 Methods]" in pkg.items[0].context_text
    assert "[context: 2 Methods]" not in pkg.items[1].context_text
    # the parent text is attached to both items for the generator
    assert pkg.items[0].parent_text == parent.text
    assert pkg.items[1].parent_text == parent.text


def test_no_parent_header_when_disabled():
    parent = _chunk("p", "parent-1", "section text")
    c1 = _chunk("p", "c1", "child", parent_id="parent-1")
    assembler = EvidenceAssembler(
        max_tokens=4000, use_parent_context=False, parent_loader=lambda pid: parent
    )
    pkg = assembler.assemble([_rc(c1)])
    assert "[context:" not in pkg.items[0].context_text


def test_missing_parent_loader_is_graceful():
    c1 = _chunk("p", "c1", "child", parent_id="nope")
    assembler = EvidenceAssembler(max_tokens=4000, use_parent_context=True)
    pkg = assembler.assemble([_rc(c1)])
    assert pkg.items[0].parent_text is None
    assert "[context:" not in pkg.items[0].context_text


def test_token_budget_enforced_and_truncation_flagged():
    counter = get_token_counter()
    t1 = "alpha " * 40   # ~240 chars
    t2 = "beta " * 40
    t3 = "gamma " * 40
    c1, c2, c3 = _chunk("p", "c1", t1), _chunk("p", "c2", t2), _chunk("p", "c3", t3)
    cost1 = counter.count(f"[1] {t1}")
    cost2 = counter.count(f"[2] {t2}")
    assembler = EvidenceAssembler(max_tokens=cost1 + cost2, use_parent_context=False)
    pkg = assembler.assemble([_rc(c1), _rc(c2), _rc(c3)])
    assert len(pkg.items) == 3            # third item is kept but degraded
    assert pkg.truncated is True
    # the third item's context was cut to the remaining budget (≈ 0)
    assert len(pkg.items[2].context_text) < len(t3)


def test_budget_fits_means_no_truncation():
    c1 = _chunk("p", "c1", "small one")
    c2 = _chunk("p", "c2", "small two")
    assembler = EvidenceAssembler(max_tokens=10000, use_parent_context=False)
    pkg = assembler.assemble([_rc(c1), _rc(c2)])
    assert pkg.truncated is False
    assert pkg.total_tokens > 0


def test_duplicate_chunks_deduplicated():
    c1 = _chunk("p", "c1", "same")
    pkg = EvidenceAssembler(max_tokens=10000, use_parent_context=False).assemble(
        [_rc(c1), _rc(c1)]
    )
    assert len(pkg.items) == 1


def test_citation_single_page_vs_range():
    single = _chunk("p", "c1", "text", page_start=3, page_end=3)
    span = _chunk("p", "c2", "text", page_start=3, page_end=5)
    pkg = EvidenceAssembler(max_tokens=10000, use_parent_context=False).assemble(
        [_rc(single), _rc(span)]
    )
    assert pkg.items[0].citation.page == 3
    assert pkg.items[0].citation.page_range == "p.3"
    assert pkg.items[1].citation.page is None
    assert pkg.items[1].citation.page_range == "p.3–5"


def test_relevance_falls_back_to_rrf_score():
    c = _chunk("p", "c1", "text")
    pkg = EvidenceAssembler(max_tokens=10000, use_parent_context=False).assemble([_rc(c, 0.42)])
    assert pkg.items[0].citation.relevance == 0.42
    # and rerank_score wins when present
    rc = RetrievedChunk(chunk=c, rrf_score=0.42, rerank_score=0.9)
    pkg2 = EvidenceAssembler(max_tokens=10000, use_parent_context=False).assemble([rc])
    assert pkg2.items[0].citation.relevance == 0.9
