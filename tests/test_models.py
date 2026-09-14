"""Unit tests for the document/chunk/retrieval models (pure data contracts)."""

from __future__ import annotations

from researchrag.models.chunk import Chunk, make_chunk_id
from researchrag.models.document import Block, BlockType, Paper, make_block_id
from researchrag.models.retrieval import (
    ImplementationSpec,
    SourceKind,
    SpecRequirement,
)


def test_make_block_id_is_stable_and_sortable():
    a = make_block_id("p1", 2, 5)
    assert a == "p1:002:0005"
    assert make_block_id("p1", 2, 5) == a


def test_make_chunk_id_is_content_addressed():
    a = make_chunk_id("p1", "c", 3, "same text")
    b = make_chunk_id("p1", "c", 3, "same text")
    c = make_chunk_id("p1", "c", 3, "other text")
    assert a == b
    assert a != c


def test_chunk_pages_display_single_and_range():
    c1 = Chunk(id="x", paper_id="p", kind="child", text="t", page_start=3, page_end=3)
    c2 = Chunk(id="y", paper_id="p", kind="child", text="t", page_start=3, page_end=5)
    assert c1.pages_display == "p.3"
    assert c2.pages_display == "p.3–5"


def test_chunk_section_display():
    c = Chunk(
        id="x", paper_id="p", kind="child", text="t",
        page_start=1, page_end=1, section_path=["2 Methods", "2.3 Generator"],
    )
    assert c.section_display == "2 Methods › 2.3 Generator"


def test_chunk_from_row_roundtrip():
    row = {
        "id": "p1:c:0001:abcd",
        "paper_id": "p1",
        "kind": "child",
        "text": "hello",
        "token_count": 2,
        "parent_id": "pp",
        "section_path": ["A", "B"],
        "section_id": 7,
        "page_start": 2,
        "page_end": 4,
        "block_ids": ["b1", "b2"],
        "chunk_type": "text",
        "metadata": {},
    }
    c = Chunk.from_row(row)
    assert c.id == "p1:c:0001:abcd"
    assert c.section_path == ["A", "B"]
    assert c.page_start == 2 and c.page_end == 4
    assert c.parent_id == "pp"


def test_chunk_provenance_record():
    c = Chunk(
        id="x", paper_id="p", kind="child", text="t",
        page_start=1, page_end=2, block_ids=["b1"],
        section_path=["S"], paper_title="Title",
    )
    prov = c.provenance()
    assert prov["chunk_id"] == "x"
    assert prov["pages"] == "p.1–2"
    assert prov["block_ids"] == ["b1"]


def test_paper_all_blocks_flattens_pages_in_order():
    from researchrag.models.document import Page

    b1 = Block(id="1", paper_id="p", block_type=BlockType.PARAGRAPH, text="a", page=1, order=1)
    b2 = Block(id="2", paper_id="p", block_type=BlockType.HEADING, text="h", page=2, order=2,
               heading_level=1, section_path=["S1"])
    paper = Paper(
        id="p", filename="x.pdf", page_count=2,
        pages=[Page(number=1, blocks=[b1]), Page(number=2, blocks=[b2])],
    )
    assert [b.id for b in paper.all_blocks()] == ["1", "2"]
    assert paper.block_count == 2
    assert paper.block_by_id("2").text == "h"


def test_spec_stats_counts_by_source_and_type():
    from researchrag.models.retrieval import RequirementType

    spec = ImplementationSpec(
        id="s", paper_id="p", title="t",
        requirements=[
            SpecRequirement(id="REQ-001", requirement="r1",
                            type=RequirementType.HYPERPARAMETER, source=SourceKind.EXPLICIT),
            SpecRequirement(id="REQ-002", requirement="r2",
                            type=RequirementType.MODEL, source=SourceKind.INFERRED),
            SpecRequirement(id="REQ-003", requirement="r3",
                            type=RequirementType.MODEL, source=SourceKind.EXPLICIT),
        ],
    )
    stats = spec.stats()
    assert stats["total"] == 3
    assert stats["by_source"] == {"explicit": 2, "inferred": 1}
    assert stats["by_type"] == {"hyperparameter": 1, "model": 2}
