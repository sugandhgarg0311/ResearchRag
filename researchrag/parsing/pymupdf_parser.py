"""Default PDF parser built on PyMuPDF (fitz).

Why PyMuPDF by default
----------------------
* Tiny single wheel, no torch/transformers — keeps the base install light.
* Fast: a 20-page paper parses in < 2 s on a laptop.
* Good enough for the structure research papers need:
  - font-size/weight based heading detection with numbering-aware levels,
  - two-column reading-order reconstruction,
  - real table detection (``page.find_tables``) with Markdown export,
  - math-font detection for formula blocks,
  - image extraction for figures with caption association,
  - footnote detection.

Limitations (documented, see PROJECT_ARCHITECTURE.md):
  * Formulas are kept as *extracted text*, not LaTeX. Variable names and
    operator glyphs are preserved well enough for sparse retrieval.
  * Two-column detection is heuristic (standard for arXiv/ICML/NeurIPS PDFs).

For higher-fidelity layouts (complex figures, multi-page tables, OCR of
scans) install the optional ``[docling]`` extra and set
``RESEARCHRAG_PARSER_BACKEND=docling``.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pymupdf
from PIL import Image

from researchrag.models.document import (
    Block,
    BlockType,
    Page,
    Paper,
    make_block_id,
)
from researchrag.parsing.base import BaseParser, ParseError

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
_APPENDIX_HEADING = re.compile(r"^appendix\s+[a-z0-9]", re.IGNORECASE)
_CAPTION_PREFIX = re.compile(r"^(figure|fig\.?|table|tab\.?)\s+(\d+[a-z]?)\s*[:.]", re.IGNORECASE)
_LIST_BULLET = re.compile(r"^[•▪◦‣·\-\–]\s+")
_LIST_PARA = re.compile(r"^\((?:[a-z]|\d{1,2})\)\s")
_MATH_FONT = re.compile(r"^(C[MRS][A-Z0-9]*|MCM|MSM|MSCR|XITSMath|STIXMath)", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

_KNOWN_TOP_SECTIONS = {
    "abstract", "introduction", "related work", "background", "method",
    "methods", "approach", "model", "architecture", "experiments",
    "results", "evaluation", "discussion", "conclusion", "conclusions",
    "references", "appendix", "acknowledgments", "acknowledgements",
    "limitations", "ethics", "bibliography", "supplementary material",
}

_MATH_CHARS = set(
    "∑∏∮θφψαβγδεζηλμνξπρστυω∇≈≤≥≠±×÷−→⇒⇔√∞∈∉⊂⊃⊆⊇∪∩∀∃¬∧∨"
    "₀₁₂₃₄₅₇₈₉¹²³⁴⁵⁶⁸⁹"
)

# Big display operators in TeX's cmex10 extension font have no usable
# ToUnicode mapping in most arXiv PDFs: the summation sign extracts as the
# letter 'X' and the product sign as 'Y' (the raw TeX slot names leaking
# through). Verified against the RAG-paper display equations by matching
# glyph bboxes to the visual operator positions (summation left of its
# z-subscript, product left of its i/N limits). Only these two slots are
# remapped — every other CMEX glyph is left untouched, and C0 control
# characters (fragments of extensible delimiters with no Unicode mapping,
# e.g. \\x00/\\x01 brace/paren pieces) are stripped so they cannot inject
# invisible junk into chunks.
_CMEX_OPERATOR_RECOVERY = {"X": "∑", "Y": "∏"}


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class PymupdfParser(BaseParser):
    name = "pymupdf"

    def __init__(self, figure_max_pixels: int = 1_600_000):
        self.figure_max_pixels = figure_max_pixels

    # ==================================================================
    # Public API
    # ==================================================================
    def parse(self, path: Path, paper_id: str, artifacts_dir: Path) -> Paper:
        path = Path(path)
        if not path.exists():
            raise ParseError(f"File not found: {path}")
        if not path.is_file():
            raise ParseError(f"Not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise ParseError(f"Unsupported file type: {path.suffix} (only PDF is parsed)")

        try:
            doc = pymupdf.open(path)
        except Exception as e:
            raise ParseError(f"Failed to open PDF: {e}") from e

        try:
            with doc:
                if doc.page_count == 0:
                    raise ParseError("PDF has no pages.")
                if doc.needs_pass:
                    raise ParseError("PDF is password-protected.")

                page_models: list[Page] = []
                section_stack: list[tuple[int, int, str]] = []  # (level, id, text)
                counters = {"next_section_id": 0, "next_order": 0}
                next_order = 0
                paper_title: str | None = None
                figure_dir = artifacts_dir / "figures"
                figure_dir.mkdir(parents=True, exist_ok=True)

                for page_index in range(doc.page_count):
                    page = doc[page_index]
                    page_no = page_index + 1
                    page_rect = page.rect
                    blocks = self._parse_page(
                        page=page,
                        doc=doc,
                        page_no=page_no,
                        page_index=page_index,
                        section_stack=section_stack,
                        counters=counters,
                        paper_id=paper_id,
                        figure_dir=figure_dir,
                        paper_title=paper_title,
                    )
                    if page_index == 0:
                        title_block = next(
                            (
                                b
                                for b in blocks
                                if b.block_type == BlockType.TITLE
                            ),
                            None,
                        )
                        if title_block is not None:
                            paper_title = title_block.text
                    # apply global ordering & ids
                    for i, b in enumerate(blocks):
                        b.order = next_order + i
                        b.id = make_block_id(paper_id, page_no, next_order + i)
                    next_order += len(blocks)

                    if not blocks:
                        raise ParseError(
                            f"No extractable text, tables or figures on page {page_no}. "
                            "The PDF may be a scanned document (enable OCR) or malformed."
                        )
                    page_models.append(
                        Page(
                            number=page_no,
                            width=page_rect.width,
                            height=page_rect.height,
                            blocks=blocks,
                        )
                    )
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"PDF parsing failed: {e}") from e

        paper = Paper(
            id=paper_id,
            filename=path.name,
            page_count=len(page_models),
            file_sha256=_sha256_of(path),
            size_bytes=path.stat().st_size,
            source_path=str(path),
            parser=self.name,
            pages=page_models,
        )
        self._fill_metadata(paper)
        self._remove_running_headers(paper)
        if paper.block_count == 0:
            raise ParseError("Document contains no blocks.")
        return paper

    def _is_running_header(self, b: Block, paper_title: str, h: float) -> bool:
        """A block that is part of the repeated paper title at the top of a
        page (arXiv appendix pages)."""
        if b.bbox is None or b.bbox[1] >= h * 0.2:
            return False
        if b.block_type not in (
            BlockType.HEADING, BlockType.PARAGRAPH, BlockType.TEXT, BlockType.TITLE
        ):
            return False
        text_l = b.text.lower()
        if text_l.startswith("appendices"):
            return True
        title_words = [w for w in paper_title.lower().split() if len(w) > 3][:8]
        if len(title_words) < 3:
            return False
        return sum(w in text_l for w in title_words) >= 2

    def _remove_running_headers(self, paper: Paper) -> None:
        """Backstop: remove any running-header blocks missed by the
        per-page pre-pass (e.g. if the title was only known later)."""
        if not paper.title or len(paper.pages) < 2:
            return
        for page in paper.pages[1:]:
            top = (page.height or 792.0) * 0.2
            page.blocks = [
                b
                for b in page.blocks
                if not self._is_running_header(b, paper.title, page.height or 792.0)
                or b.bbox is None
                or b.bbox[1] >= top
            ]

    # ==================================================================
    # Per-page pipeline
    # ==================================================================
    def _parse_page(
        self,
        page,
        doc,
        page_no: int,
        page_index: int,
        section_stack: list,
        counters: dict,
        paper_id: str,
        figure_dir: Path,
        paper_title: str | None = None,
    ) -> list[Block]:
        page_rect = page.rect
        midx = page_rect.width / 2
        h = page_rect.height

        tables = self._find_tables(page)
        images = self._find_images(page, doc, paper_id, page_no, figure_dir)

        # ---- pass 1: raw text items -------------------------------------
        text_items = self._extract_text_items(page)
        text_items = [t for t in text_items if not self._is_page_artifact(t.text)]
        text_items = self._join_hyphenated(text_items)
        body_size = self._estimate_body_size(text_items)

        raw_blocks: list[Block] = []

        # title candidate (page 1 only)
        title_items: list = []
        if page_index == 0:
            title_items = self._extract_title_items(text_items, body_size, h)
        title_texts = [t.text for t in title_items]

        for item in text_items:
            if item in title_items:
                continue
            if self._inside_table(item.bbox, tables):
                continue
            block = self._classify_text_item(item, body_size, h, page_index)
            if block is not None:
                raw_blocks.append(block)

        if title_items:
            t = Block(
                id="", paper_id=paper_id, block_type=BlockType.TITLE,
                text=" ".join(title_texts), page=page_no, order=-1,
                bbox=list(title_items[0].bbox),
            )
            raw_blocks.append(t)

        # tables & figures
        for t in tables:
            block = Block(
                id="", paper_id=paper_id, block_type=BlockType.TABLE,
                text=t["text"], page=page_no, order=-1,
                bbox=list(t["bbox"]), table_rows=t["rows"], table_cols=t["cols"],
            )
            if t.get("markdown"):
                block.metadata["table_markdown"] = t["markdown"]
            raw_blocks.append(block)
        for img in images:
            raw_blocks.append(
                Block(
                    id="", paper_id=paper_id, block_type=BlockType.FIGURE,
                    text="", page=page_no, order=-1,
                    bbox=list(img["bbox"]), image_path=img["image_path"],
                )
            )

        # fix physical page on text-classified blocks
        for b in raw_blocks:
            b.page = page_no

        # ---- pass 2: reading-order sort ----------------------------------
        def sort_key(b: Block):
            x0, y0, x1, y1 = (b.bbox or [0, 0, 0, 0])
            w = x1 - x0
            cx = (x0 + x1) / 2
            crosses_center = x0 < midx < x1
            is_full = w >= page_rect.width * 0.85
            if crosses_center or is_full:
                if y0 < h * 0.25:
                    return (0, y0)  # header region
                if y0 > h * 0.88:
                    return (3, y0)  # footer region
                return (1 if cx < midx else 2, y0)
            return (1 if cx < midx else 2, y0)

        ordered = sorted(raw_blocks, key=sort_key)

        # ---- remove running headers (repeated title) BEFORE the section
        #      stack is built, so they can never corrupt the hierarchy ----
        if paper_title and page_index > 0:
            ordered = [
                b
                for b in ordered
                if not self._is_running_header(b, paper_title, h)
            ]

        # ---- merge shattered display equations ---------------------------
        # PyMuPDF emits one dict block per visual line/run, so a display
        # equation arrives as several FORMULA blocks (LHS, big operators,
        # sub/superscript limits, RHS terms). Merge consecutive FORMULA
        # blocks that touch into one complete equation block.
        ordered = self._merge_formula_runs(ordered)

        # ---- pass 3: assign section hierarchy in reading order -----------
        for b in ordered:
            b.paper_id = paper_id
            if b.block_type == BlockType.HEADING:
                level = b.heading_level or 1
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                counters["next_section_id"] += 1
                sec_id = counters["next_section_id"]
                section_stack.append((level, sec_id, b.text))
                b.section_id = sec_id
                b.section_path = [s[2] for s in section_stack]
            elif b.block_type in (BlockType.TITLE,):
                b.section_path = []
                b.section_id = None
            else:
                b.section_path = [s[2] for s in section_stack]
                b.section_id = section_stack[-1][1] if section_stack else None
            b.metadata["section_path"] = list(b.section_path)

        # ---- pass 4: link captions to figures/tables ----------------------
        self._link_captions(ordered)
        return ordered

    @staticmethod
    def _near_asset(caption: Block, blocks: list[Block]) -> bool:
        if caption.bbox is None:
            return False
        c_y0, c_y1 = caption.bbox[1], caption.bbox[3]
        for b in blocks:
            if b is caption or b.bbox is None:
                continue
            if b.block_type not in (BlockType.FIGURE, BlockType.TABLE):
                continue
            gap = c_y0 - b.bbox[3] if c_y0 >= b.bbox[3] else b.bbox[1] - c_y1
            if 0 <= gap <= 40.0:
                return True
        return False

    def _link_captions(self, blocks: list[Block]) -> None:
        """Attach 'Figure N: ...' / 'Table N: ...' captions to their assets."""
        for b in blocks:
            if b.block_type not in (BlockType.FIGURE, BlockType.TABLE):
                continue
            by, = (b.bbox[1],) if b.bbox else (0,)
            by_end = b.bbox[3] if b.bbox else 0
            best = None
            best_dist = 40.0  # max gap in points
            for c in blocks:
                if c.block_type != BlockType.PARAGRAPH or c.bbox is None:
                    continue
                m = _CAPTION_PREFIX.match(c.text.strip())
                if not m:
                    continue
                c_y0 = c.bbox[1]
                dist = c_y0 - by_end if c_y0 >= by_end else by - c.bbox[3]
                if 0 <= dist <= best_dist:
                    best, best_dist = c, dist
            if best is not None:
                b.caption = best.text.strip()
                best.block_type = BlockType.CAPTION
                best.metadata["for_block"] = b.id
                best.metadata["caption_kind"] = "figure" if b.block_type == BlockType.FIGURE else "table"

        # Figure captions whose asset is vector graphics (not an extracted
        # raster image and not a ruled table): synthesize a FIGURE block so
        # the figure remains first-class with its caption + section context.
        inserts: list[tuple[int, Block]] = []
        for i, c in enumerate(blocks):
            if c.block_type != BlockType.PARAGRAPH:
                continue
            if len(c.text.strip()) > 450:  # in-text discussion, not a caption
                continue
            m = _CAPTION_PREFIX.match(c.text.strip())
            if not m or m.group(1).lower().startswith("tab"):
                continue
            if self._near_asset(c, blocks):
                continue
            fig = Block(
                id=c.id + ":fig",
                paper_id=c.paper_id,
                block_type=BlockType.FIGURE,
                text=c.text,
                page=c.page,
                order=c.order,
                bbox=list(c.bbox) if c.bbox else None,
                caption=c.text.strip(),
                section_path=list(c.section_path),
                section_id=c.section_id,
            )
            fig.metadata["section_path"] = list(c.section_path)
            fig.metadata["vector_only"] = True
            inserts.append((i, fig))
            c.block_type = BlockType.CAPTION
            c.metadata["for_block"] = fig.id
            c.metadata["caption_kind"] = "figure"
        for offset, (i, fig) in enumerate(inserts):
            blocks.insert(i + offset, fig)

    # ==================================================================
    # Formula-run merging
    # ==================================================================
    @staticmethod
    def _merge_formula_runs(blocks: list[Block]) -> list[Block]:
        """Merge consecutive FORMULA blocks that touch into one equation.

        A display equation typically arrives shattered: LHS line, big
        operators (∑/∏), sub/superscript limit lines, RHS terms — each its
        own dict block. Merging rule (conservative): same page order,
        both FORMULA, both with bboxes, vertical gap <= 30pt, and
        horizontal intervals overlapping or within 80pt. Anything else
        (a paragraph, heading, table between them, or a far-away block)
        breaks the run. Text is space-joined; bbox is the union.
        """
        merged: list[Block] = []
        for b in blocks:
            prev = merged[-1] if merged else None
            if (
                prev is not None
                and prev.block_type == BlockType.FORMULA
                and b.block_type == BlockType.FORMULA
                and prev.bbox is not None
                and b.bbox is not None
                and PymupdfParser._formula_touch(prev.bbox, b.bbox)
            ):
                prev.text = (prev.text + " " + b.text).strip()
                prev.bbox = [
                    min(prev.bbox[0], b.bbox[0]),
                    min(prev.bbox[1], b.bbox[1]),
                    max(prev.bbox[2], b.bbox[2]),
                    max(prev.bbox[3], b.bbox[3]),
                ]
                prev.metadata["merged_formula_parts"] = (
                    prev.metadata.get("merged_formula_parts", 1) + 1
                )
            else:
                merged.append(b)
        return merged

    @staticmethod
    def _formula_touch(a: list[float], b: list[float]) -> bool:
        v_gap = b[1] - a[3]  # next top minus prev bottom (<=0 overlaps)
        if v_gap > 30.0:
            return False
        h_overlap = min(a[2], b[2]) - max(a[0], b[0])
        if h_overlap >= 0:
            return True
        return -h_overlap <= 80.0

    # ==================================================================
    # Title / authors / abstract
    # ==================================================================
    @staticmethod
    def _is_page_artifact(text: str) -> bool:
        """arXiv watermarks, page numbers, etc."""
        if re.match(r"^arXiv:\d{4}\.\d{4,5}v\d+", text):
            return True
        if re.fullmatch(r"\d{1,4}", text.strip()):  # bare page number
            return True
        return False

    @staticmethod
    def _is_affiliation_like(text: str) -> bool:
        if re.search(r"(University|Institute|College|Laboratory|Research|Academy)", text):
            return True
        if any(c in text for c in "†‡§⋆●") and len(text) < 20:
            return True
        return False

    def _extract_title_items(self, text_items: list, body_size: float, h: float) -> list:
        """Collect the (possibly multi-line) title: the largest-font block(s)
        at the very top of page 1, excluding affiliations/author lines."""
        candidates = [
            t
            for t in text_items
            if t.size >= body_size * 1.12
            and t.bbox[1] < h * 0.35
            and 8 <= len(t.text) <= 300
            and not self._is_affiliation_like(t.text)
            and not _EMAIL.search(t.text)
            and not self._has_affiliation_marks(t.text)
        ]
        if not candidates:
            return []
        candidates.sort(key=lambda t: t.bbox[1])
        top = candidates[0]
        run = [top]
        prev = top
        for t in candidates[1:]:
            # continue the run while the block is close below the previous one
            # AND has a comparable font size (title lines match)
            if t.bbox[1] - prev.bbox[3] < body_size * 4 and t.size >= top.size * 0.9:
                run.append(t)
                prev = t
            else:
                break
        return run

    @staticmethod
    def _has_affiliation_marks(text: str) -> bool:
        marks = sum(text.count(c) for c in "†‡§⋆¶●")
        return marks >= 2

    @staticmethod
    def _join_hyphenated(items: list) -> list:
        """Re-join words split across blocks by a line-break hyphen
        (e.g. 'Margin-' + 'alize' -> 'Marginalize')."""
        if not items:
            return items
        out = []
        for item in items:
            if (
                out
                and out[-1].text.endswith("-")
                and not out[-1].text.endswith("--")
                and item.text
                and item.text[0].islower()
                and not item.is_math
                and not out[-1].is_math
                and abs(item.size - out[-1].size) < 1.0
                and item.bbox[1] - out[-1].bbox[3] < 15
            ):
                prev = out[-1]
                prev.text = prev.text[:-1] + item.text
                prev.bbox = [
                    min(prev.bbox[0], item.bbox[0]),
                    prev.bbox[1],
                    max(prev.bbox[2], item.bbox[2]),
                    item.bbox[3],
                ]
            else:
                out.append(item)
        return out

    def _fill_metadata(self, paper: Paper) -> None:
        first = paper.pages[0] if paper.pages else None
        if not first:
            return
        midx = (first.width or 612.0) / 2

        title_text = None
        author_lines: list[str] = []
        abstract_parts: list[str] = []

        # --- title & authors ------------------------------------------------
        title_block = next(
            (b for b in first.blocks if b.block_type == BlockType.TITLE), None
        )
        if title_block is not None:
            title_text = title_block.text
        for b in first.blocks:
            if b.block_type not in (BlockType.PARAGRAPH, BlockType.TEXT):
                continue
            if b.bbox is None or b.bbox[1] > (first.height or 792.0) * 0.45:
                continue
            text = b.text
            # author lines carry affiliation marks or look like name lists
            has_marks = any(c in text for c in "†‡§¶●⋆")
            name_list = bool(re.match(r"^[\w'’.-]+(?:\s+[\w'’.-]+){1,3},\s*[\w'’.-]+", text))
            if has_marks or name_list:
                author_lines.append(text)

        # --- abstract: column-aware (two-column pages break reading order
        #     assumptions) ------------------------------------------------
        abs_heading = next(
            (
                b
                for b in first.blocks
                if b.block_type == BlockType.HEADING
                and b.text.strip().lower() == "abstract"
            ),
            None,
        )
        if abs_heading is not None and abs_heading.bbox:
            page_w = first.width or 612.0
            abs_cx = (abs_heading.bbox[0] + abs_heading.bbox[2]) / 2
            centered = abs(abs_cx - midx) < 30
            # next structural heading below the abstract heading
            next_heading_y = float("inf")
            for b in first.blocks:
                if b is abs_heading or b.bbox is None:
                    continue
                if b.block_type == BlockType.HEADING and b.bbox[1] > abs_heading.bbox[1]:
                    next_heading_y = min(next_heading_y, b.bbox[1])
            for b in first.blocks:
                if b.bbox is None or b is abs_heading:
                    continue
                if b.block_type not in (BlockType.PARAGRAPH, BlockType.TEXT):
                    continue
                y = b.bbox[1]
                if not (abs_heading.bbox[1] + 10 < y < next_heading_y):
                    continue
                if centered:
                    # abstract spans the center: keep centered / wide blocks
                    bcx = (b.bbox[0] + b.bbox[2]) / 2
                    bwidth = b.bbox[2] - b.bbox[0]
                    if abs(bcx - midx) < 40 or bwidth > page_w * 0.55:
                        abstract_parts.append(b.text)
                else:
                    in_col = (b.bbox[0] + b.bbox[2]) / 2 < midx == (abs_cx < midx)
                    if in_col:
                        abstract_parts.append(b.text)

        paper.title = title_text
        if title_text is None:
            for b in first.blocks:
                if b.block_type in (BlockType.PARAGRAPH, BlockType.TEXT):
                    paper.title = b.text
                    break
        paper.authors = self._parse_authors(author_lines)
        if abstract_parts:
            paper.abstract = " ".join(sorted(abstract_parts))
        paper.metadata["block_counts"] = self._count_types(paper)

    @staticmethod
    def _parse_authors(lines: list[str]) -> list[str]:
        """Per-line parsing so affiliation lines never contaminate names."""
        names: list[str] = []
        for line in lines:
            if _EMAIL.search(line):
                continue
            if re.search(
                r"(University|Institute|College|Laboratory|Research|Academy|Department)",
                line,
            ):
                continue
            cleaned = re.sub(r"[†‡§¶●⋆]", " ", line)
            for part in cleaned.split(","):
                part = re.sub(r"\s+", " ", part).strip()
                if part and re.fullmatch(r"[\w'’.-]+(?:\s+[\w'’.-]+){0,3}", part):
                    names.append(part)
        # de-duplicate, preserve order
        seen = set()
        out = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out[:40]

    @staticmethod
    def _count_types(paper: Paper) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in paper.all_blocks():
            counts[b.block_type.value] = counts.get(b.block_type.value, 0) + 1
        return counts

    # ==================================================================
    # Text extraction
    # ==================================================================
    class _TextItem:
        __slots__ = ("text", "bbox", "size", "bold", "full_width", "is_math", "math_font")

        def __init__(self, text, bbox, size, bold, full_width, is_math, math_font=False):
            self.text = text
            self.bbox = bbox  # [x0, y0, x1, y1]
            self.size = size
            self.bold = bold
            self.full_width = full_width
            self.is_math = is_math
            self.math_font = math_font  # any span set in a TeX math font

    @staticmethod
    def _recover_span_text(font: str, text: str) -> str:
        """Recover usable text from a raw PDF span.

        * CMEX10 big operators (verified slots only): 'X' -> ∑, 'Y' -> ∏.
        * C0 control characters (unmapped delimiter fragments) are stripped.
        """
        if font.upper().startswith("CMEX"):
            text = "".join(_CMEX_OPERATOR_RECOVERY.get(c, c) for c in text)
        return "".join(c for c in text if ord(c) >= 32 or c in "\t")

    def _extract_text_items(self, page) -> list:
        d = page.get_text("dict")
        items: list[self._TextItem] = []
        page_rect = page.rect
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = list(block["bbox"])
            width = bbox[2] - bbox[0]
            full_width = width >= page_rect.width * 0.85

            line_texts = []
            max_size = 0.0
            bold_chars = 0
            math_chars = 0
            math_font = False
            char_total = 0
            for line in block.get("lines", []):
                line_parts = []
                for span in line.get("spans", []):
                    raw = span.get("text", "")
                    font = span.get("font", "")
                    text = self._recover_span_text(font, raw)
                    if _MATH_FONT.match(font):
                        math_font = True
                    if not text.strip():
                        if line_parts and not line_parts[-1].endswith(" "):
                            line_parts.append(" ")
                        continue
                    line_parts.append(text)
                    n = max(len(text), 1)
                    max_size = max(max_size, span.get("size", 0) or 0)
                    char_total += n
                    flags = span.get("flags", 0)
                    font = span.get("font", "")
                    if (flags & 16) or re.search(r"(Bold|Medi|Black|SemiBold)", font, re.I):
                        bold_chars += n
                    if _MATH_FONT.match(font) or self._math_glyph_ratio(text) > 0.5:
                        math_chars += n
                line_text = "".join(line_parts).strip()
                if line_text:
                    line_texts.append(line_text)

            if not line_texts:
                continue
            text = " ".join(line_texts)
            if char_total == 0:
                continue
            is_math = (math_chars / max(char_total, 1)) > 0.4
            items.append(
                self._TextItem(
                    text=text,
                    bbox=bbox,
                    size=max_size,
                    bold=bold_chars / max(char_total, 1) >= 0.5,
                    full_width=full_width,
                    is_math=is_math,
                    math_font=math_font,
                )
            )
        return items

    @staticmethod
    def _math_glyph_ratio(text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for c in text if c in _MATH_CHARS) / len(text)

    @staticmethod
    def _estimate_body_size(items: list) -> float:
        counts: dict[float, int] = {}
        for it in items:
            if it.is_math or it.full_width:
                continue
            key = round(it.size, 1)
            counts[key] = counts.get(key, 0) + len(it.text)
        if not counts:
            return 10.0
        return max(counts.items(), key=lambda kv: kv[1])[0]

    # ==================================================================
    # Classification
    # ==================================================================
    def _classify_text_item(self, item, body_size: float, page_h: float, page_index: int):
        text = item.text.strip()
        if not text:
            return None
        bbox = item.bbox

        if item.is_math and (len(text) >= 2 or item.math_font):
            # Single-character items are kept only when set in a TeX math
            # font (e.g. a lone CMEX display operator): an isolated math
            # glyph is virtually always an equation fragment, and dropping
            # it deletes operators (the old len>=2 gate turned a lone ∑
            # into a stray "X" paragraph).
            b = self._block(BlockType.FORMULA, text, bbox)
            b.metadata["is_math"] = True
            return b

        if (
            bbox[1] > page_h * 0.88
            and item.size < body_size * 0.98
            and not item.bold
        ):
            return self._block(BlockType.FOOTNOTE, text, bbox)

        level = self._heading_level(item, body_size, page_index, page_h)
        if level is not None:
            b = self._block(BlockType.HEADING, text, bbox)
            b.heading_level = level
            return b

        if _LIST_BULLET.match(text) or _LIST_PARA.match(text):
            return self._block(BlockType.LIST_ITEM, text, bbox)

        return self._block(BlockType.PARAGRAPH, text, bbox)

    def _heading_level(self, item, body_size: float, page_index: int, page_h: float) -> int | None:
        text = item.text.strip()
        if len(text) > 130 or item.is_math:
            return None
        # short formula fragments like 'q(x)' are not headings
        if (
            " " not in text
            and len(text) < 10
            and re.search(r"[()\[\]{}=+*/|&^_]", text)
        ):
            return None
        # table rows: multiple decimal numbers -> not a heading
        if len(re.findall(r"\d+\.\d+", text)) >= 2:
            return None
        lowered = text.lower().strip().rstrip(":")
        is_known = lowered in _KNOWN_TOP_SECTIONS or bool(_APPENDIX_HEADING.match(text))
        m = _NUMBERED_HEADING.match(text)
        if m:
            # '10 20 30 40 50 K ...' (axis labels) must not match: the number
            # must be followed by a word (the heading title).
            rest = text[m.end():].strip()
            if rest[:1].isalpha():
                return min(m.group(1).count(".") + 1, 4)
            return None
        if is_known:
            return 1
        # Page-1 front matter (title/authors region): suppress unnumbered
        # headings above the top third so author lines don't become sections.
        if page_index == 0 and item.bbox[1] < page_h * 0.35:
            return None
        if item.bold or item.size >= body_size * 1.15:
            if item.size >= body_size * 1.2:
                return 1
            return 2
        return None

    @staticmethod
    def _block(block_type: BlockType, text: str, bbox) -> Block:
        return Block(
            id="",
            paper_id="",
            block_type=block_type,
            text=text,
            page=0,
            order=-1,
            bbox=list(bbox) if bbox else None,
        )

    # ==================================================================
    # Tables / figures
    # ==================================================================
    def _find_tables(self, page) -> list:
        """Detect real (ruled) tables with the high-precision 'lines_strict'
        strategy, then extract their text row-by-row from the underlying text
        layer (PyMuPDF's row inference is unreliable when tables lack inner
        horizontal rules; the raw lines are what retrieval and LLMs need)."""
        try:
            found = page.find_tables(strategy="lines_strict")
        except Exception:
            found = None
        tables = []
        if found is not None:
            for t in found.tables:
                bbox = list(t.bbox)
                lines = self._lines_in_bbox(page, bbox, pad=2.0)
                if not lines:
                    continue
                text = "\n".join(lines)
                try:
                    markdown = t.to_markdown().strip()
                except Exception:
                    markdown = ""
                tables.append(
                    {
                        "bbox": bbox,
                        "rows": len(lines),
                        "cols": t.col_count,
                        "text": text,
                        "markdown": markdown,
                    }
                )
        return tables

    @staticmethod
    def _lines_in_bbox(page, bbox, pad: float = 0.0) -> list[str]:
        """Text lines inside bbox, clustered into visual rows (by y) and
        ordered (row, x). Cells on the same visual row are joined by spaces."""
        x0, y0, x1, y1 = bbox
        d = page.get_text("dict")
        rows: list[tuple[float, float, str]] = []
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                lb = line["bbox"]
                cx = (lb[0] + lb[2]) / 2
                cy = (lb[1] + lb[3]) / 2
                if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
                    text = "".join(s.get("text", "") for s in line["spans"]).strip()
                    if text:
                        rows.append((cy, cx, text))
        rows.sort(key=lambda r: (r[0], r[1]))

        lines: list[str] = []
        current_y: float | None = None
        current: list[tuple[float, str]] = []
        for cy, cx, text in rows:
            if current_y is not None and abs(cy - current_y) > 3.5:
                lines.append(" ".join(t for _, t in sorted(current)))
                current = []
            current_y = cy if current_y is None else current_y
            current.append((cx, text))
        if current:
            lines.append(" ".join(t for _, t in sorted(current)))
        return lines

    @staticmethod
    def _inside_table(bbox, tables, pad: float = 3.0) -> bool:
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for t in tables:
            tb = t["bbox"]
            if tb[0] - pad <= cx <= tb[2] + pad and tb[1] - pad <= cy <= tb[3] + pad:
                return True
        return False

    def _find_images(self, page, doc, paper_id: str, page_no: int, figure_dir: Path) -> list:
        try:
            infos = page.get_image_info(hashes=False)
        except Exception:
            infos = []
        images = []
        for idx, info in enumerate(infos):
            bbox = list(info.get("bbox", (0, 0, 0, 0)))
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 2000:
                continue
            image_path = None
            try:
                xref = info.get("xref", 0)
                if xref:
                    ext = doc.extract_image(xref)
                    if ext:
                        img = Image.open(io.BytesIO(ext["image"]))
                        w, hgt = img.size
                        if w * hgt > self.figure_max_pixels:
                            scale = (self.figure_max_pixels / (w * hgt)) ** 0.5
                            img = img.resize((max(1, int(w * scale)), max(1, int(hgt * scale))))
                        out = figure_dir / f"p{page_no:02d}_{idx:02d}_{paper_id[-6:]}.png"
                        if img.mode not in ("RGB", "L"):
                            img = img.convert("RGB")
                        img.save(out, format="PNG")
                        image_path = str(out)
            except Exception:
                image_path = None
            images.append({"bbox": bbox, "image_path": image_path})
        return images
