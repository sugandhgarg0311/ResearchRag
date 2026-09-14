"""Unit tests for formula extraction completeness.

Covers the three parser fixes for shattered display equations:
1. CMEX big-operator recovery ('X' -> ∑, 'Y' -> ∏) + C0 control stripping.
2. Single-character math-font items are kept as FORMULA (the old len>=2
   gate turned a lone ∑ into a stray "X" paragraph).
3. Adjacent FORMULA blocks merge into one complete equation block.
"""

from __future__ import annotations

from researchrag.models.document import Block, BlockType
from researchrag.parsing.pymupdf_parser import PymupdfParser as P


def _fblock(order: int, text: str, bbox: list[float]) -> Block:
    return Block(
        id=f"p:001:{order:04d}",
        paper_id="p",
        block_type=BlockType.FORMULA,
        text=text,
        page=1,
        order=order,
        bbox=bbox,
    )


def _para(order: int, text: str) -> Block:
    return Block(
        id=f"p:001:{order:04d}",
        paper_id="p",
        block_type=BlockType.PARAGRAPH,
        text=text,
        page=1,
        order=order,
        bbox=[0, 0, 100, 12],
    )


# --- 1. operator recovery ------------------------------------------------

def test_cmex_summation_recovered():
    assert P._recover_span_text("CMEX10", "X") == "∑"


def test_cmex_product_recovered():
    assert P._recover_span_text("CMEX10", "Y") == "∏"


def test_cmex_other_glyphs_untouched():
    assert P._recover_span_text("CMEX10", "()") == "()"


def test_non_cmex_fonts_untouched():
    assert P._recover_span_text("CMR10", "X") == "X"
    assert P._recover_span_text("CMMI10", "Y") == "Y"


def test_control_chars_stripped():
    assert P._recover_span_text("CMEX10", "\x00a\x01") == "a"


# --- 2. single-character math gate ----------------------------------------

def _classify(text: str, math_font: bool):
    parser = P()
    item = parser._TextItem(
        text=text, bbox=[0, 0, 50, 12], size=10.0,
        bold=False, full_width=False, is_math=True, math_font=math_font,
    )
    return parser._classify_text_item(item, body_size=10.0, page_h=792.0, page_index=1)


def test_lone_cmex_operator_is_formula():
    b = _classify("∑", math_font=True)
    assert b is not None and b.block_type == BlockType.FORMULA


def test_lone_non_math_letter_is_not_formula():
    b = _classify("a", math_font=False)
    assert b is not None and b.block_type == BlockType.PARAGRAPH


# --- 3. formula-run merging ------------------------------------------------

def test_touching_formulas_merge():
    blocks = [
        _fblock(0, "p(y|x) =", [180, 372, 250, 383]),
        _fblock(1, "N ∏", [257, 362, 269, 380]),
        _fblock(2, "∑", [292, 370, 307, 380]),
    ]
    out = P._merge_formula_runs(blocks)
    assert len(out) == 1
    assert out[0].text == "p(y|x) = N ∏ ∑"
    assert out[0].bbox == [180, 362, 307, 383]
    assert out[0].metadata["merged_formula_parts"] == 3


def test_paragraph_breaks_run():
    blocks = [_fblock(0, "a", [0, 0, 50, 12]), _para(1, "words"), _fblock(2, "b", [0, 20, 50, 32])]
    assert len(P._merge_formula_runs(blocks)) == 3


def test_distant_formulas_do_not_merge():
    blocks = [_fblock(0, "a", [0, 0, 50, 12]), _fblock(1, "b", [0, 200, 50, 212])]
    assert len(P._merge_formula_runs(blocks)) == 2


def test_far_apart_columns_do_not_merge():
    blocks = [_fblock(0, "a", [0, 0, 50, 12]), _fblock(1, "b", [400, 0, 450, 12])]
    assert len(P._merge_formula_runs(blocks)) == 2


def test_missing_bbox_never_merges():
    a = _fblock(0, "a", [0, 0, 50, 12])
    b = _fblock(1, "b", [0, 14, 50, 26])
    b.bbox = None
    assert len(P._merge_formula_runs([a, b])) == 2
