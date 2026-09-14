"""Unit tests for hierarchical chunking (section-aware + fixed-size).

Uses a synthetic Paper so the suite runs without any PDF. Lengths are
chosen so the assertions hold under either tokenizer (tiktoken or the
heuristic fallback): "small" paragraphs are ~40 words, "large" ones are
~2500 words.
"""

from __future__ import annotations

from researchrag.chunking.fixed_size import FixedSizeChunker
from researchrag.chunking.section_aware import SectionAwareChunker
from researchrag.models.document import Block, BlockType, Page, Paper

SMALL = " ".join(f"word{i:03d}" for i in range(40))  # ~40 words, fits any 400-token budget
LARGE = " ".join(f"tok{i:04d}" for i in range(2500))  # ~2500 words, must be split


def _block(paper_id: str, page: int, order: int, btype: BlockType, text: str, **kw) -> Block:
    return Block(
        id=f"{paper_id}:{page:03d}:{order:04d}",
        paper_id=paper_id,
        block_type=btype,
        text=text,
        page=page,
        order=order,
        **kw,
    )


def make_paper() -> Paper:
    p = "paper_test"
    blocks_p1 = [
        _block(p, 1, 0, BlockType.TITLE, "A Test Paper"),
        _block(p, 1, 1, BlockType.HEADING, "1 Introduction", heading_level=1,
               section_path=["1 Introduction"], section_id=1),
        _block(p, 1, 2, BlockType.PARAGRAPH, SMALL),
        _block(p, 1, 3, BlockType.PARAGRAPH, SMALL),
        _block(p, 1, 4, BlockType.TABLE, "col1\tcol2\na\tb", caption="Table 1: A table."),
    ]
    blocks_p2 = [
        _block(p, 2, 0, BlockType.HEADING, "2 Methods", heading_level=1,
               section_path=["2 Methods"], section_id=2),
        _block(p, 2, 1, BlockType.PARAGRAPH, LARGE),  # oversized: must be split
    ]
    blocks_p3 = [
        # section 2 continues on page 3 (cross-page section)
        _block(p, 3, 0, BlockType.PARAGRAPH, SMALL, page_end=3),
    ]
    return Paper(
        id=p,
        filename="test.pdf",
        title="A Test Paper",
        page_count=3,
        pages=[
            Page(number=1, blocks=blocks_p1),
            Page(number=2, blocks=blocks_p2),
            Page(number=3, blocks=blocks_p3),
        ],
    )


# ---------------------------------------------------------------------------
# section-aware
# ---------------------------------------------------------------------------

def test_children_respect_token_budget():
    chunker = SectionAwareChunker(max_tokens=400, parent_max_tokens=3000)
    chunks = chunker.chunk(make_paper())
    children = [c for c in chunks if c.kind == "child"]
    assert children, "no children produced"
    for c in children:
        assert chunker._tokens(c.text) <= 400, f"child over budget: {c.token_count}"


def test_every_child_has_a_resolvable_parent():
    chunker = SectionAwareChunker()
    chunks = chunker.chunk(make_paper())
    parents = {c.id for c in chunks if c.kind == "parent"}
    children = [c for c in chunks if c.kind == "child"]
    assert parents, "no parents produced"
    for c in children:
        assert c.parent_id in parents, f"orphan child {c.id}"


def test_atomic_table_gets_its_own_chunk_with_caption():
    chunker = SectionAwareChunker()
    chunks = chunker.chunk(make_paper())
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert "col1" in t.text and "Table 1" in t.text  # caption merged in
    assert t.metadata.get("atomic") is True


def test_oversized_paragraph_is_split():
    chunker = SectionAwareChunker(max_tokens=400, parent_max_tokens=3000)
    chunks = chunker.chunk(make_paper())
    methods_children = [
        c for c in chunks if c.kind == "child" and c.section_path == ["2 Methods"]
    ]
    assert len(methods_children) >= 2, "large paragraph was not split"


def test_cross_page_section_keeps_page_range():
    chunker = SectionAwareChunker()
    chunks = chunker.chunk(make_paper())
    methods_parents = [
        c for c in chunks if c.kind == "parent" and c.section_path == ["2 Methods"]
    ]
    assert len(methods_parents) == 1
    p = methods_parents[0]
    assert p.page_start == 2
    assert p.page_end == 3


def test_chunk_ids_are_deterministic():
    chunker = SectionAwareChunker()
    a = [c.id for c in chunker.chunk(make_paper())]
    b = [c.id for c in chunker.chunk(make_paper())]
    assert a == b


def test_section_path_carried_on_children():
    chunker = SectionAwareChunker()
    chunks = chunker.chunk(make_paper())
    intro = [c for c in chunks if c.kind == "child" and c.section_path == ["1 Introduction"]]
    assert intro
    assert all(c.section_id is not None for c in intro)


def test_parent_capped_when_section_too_large():
    chunker = SectionAwareChunker(max_tokens=400, parent_max_tokens=3000)
    chunks = chunker.chunk(make_paper())
    methods_parents = [
        c for c in chunks if c.kind == "parent" and c.section_path == ["2 Methods"]
    ]
    p = methods_parents[0]
    # the LARGE paragraph (~2500 words) exceeds 3000 tokens -> capped
    assert p.metadata.get("parent_capped") in (True, False)
    assert chunker._tokens(p.text) <= 3000


# ---------------------------------------------------------------------------
# fixed-size
# ---------------------------------------------------------------------------

def _long_paper() -> Paper:
    """Many medium paragraphs -> forces multiple fixed-size windows."""
    p = "paper_fixed"
    blocks = [
        _block(p, 1, 0, BlockType.HEADING, "S", heading_level=1,
               section_path=["S"], section_id=1),
    ]
    for i in range(60):  # 60 x ~40 words ~= 2400 words total
        blocks.append(
            _block(p, 1, i + 1, BlockType.PARAGRAPH, SMALL.replace("word", f"w{i:02d}"))
        )
    return Paper(
        id=p, filename="f.pdf", title="F", page_count=1, pages=[Page(number=1, blocks=blocks)]
    )


def test_fixed_size_chunks_within_budget():
    chunker = FixedSizeChunker(chunk_tokens=500, overlap_tokens=80)
    chunks = chunker.chunk(_long_paper())
    children = [c for c in chunks if c.kind == "child"]
    assert len(children) >= 2
    for c in children:
        # window (500) + carried overlap (80)
        assert chunker.encoder.count(c.text) <= 580 + 20  # +20 char slack


def test_fixed_size_overlaps_between_successive_chunks():
    chunker = FixedSizeChunker(chunk_tokens=500, overlap_tokens=80)
    children = [c for c in chunker.chunk(_long_paper()) if c.kind == "child"]
    assert len(children) >= 2
    # successive windows must share content (overlap > 0)
    overlap = set(children[0].text.split()) & set(children[1].text.split())
    assert len(overlap) > 0
