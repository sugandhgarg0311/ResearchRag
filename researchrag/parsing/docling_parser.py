"""Optional high-fidelity PDF parser backed by IBM Docling.

Docling's DeepDoc layout model outperforms font heuristics on complex
layouts (multi-page tables, atypical heading styles, scanned pages with
OCR). It is an **optional extra** because it pulls in torch + deep models
(~1.5 GB installed, slower on CPU):

    pip install "researchrag[docling]"
    export RESEARCHRAG_PARSER_BACKEND=docling

The adapter maps Docling items onto the same canonical ``Paper`` model as
the PyMuPDF parser, so the rest of the pipeline is parser-agnostic.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from researchrag.models.document import Block, BlockType, Page, Paper, make_block_id
from researchrag.parsing.base import BaseParser, ParseError


class DoclingParser(BaseParser):
    name = "docling"

    def supported(self) -> bool:
        try:
            import docling  # noqa: F401
            return True
        except Exception:
            return False

    def parse(self, path: Path, paper_id: str, artifacts_dir: Path) -> Paper:
        if not self.supported():
            raise ParseError(
                "Docling backend is not installed. Run: pip install 'researchrag[docling]'"
            )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import (
            CodeItem,
            FormulaItem,
            PictureItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
            TitleItem,
        )

        path = Path(path)
        if not path.exists():
            raise ParseError(f"File not found: {path}")

        options = PdfPipelineOptions()
        options.do_table_structure = True
        options.do_formula_enrichment = True
        options.generate_picture_images = True
        options.images_scale = 2.0

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        try:
            result = converter.convert(path)
            doc = result.document
        except Exception as e:
            raise ParseError(f"Docling conversion failed: {e}") from e

        figure_dir = artifacts_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)

        section_stack: list[tuple[int, int, str]] = []
        next_section_id = 0
        next_order = 0
        pages: list[Page] = []
        page_no = 0

        for item, _level in doc.iterate_items():
            prov = getattr(item, "prov", None)
            if not prov:
                continue
            new_page = int(prov[0].page_no)
            if not pages or pages[-1].number != new_page:
                page_no = new_page
                try:
                    rect = doc.page_size(new_page) if hasattr(doc, "page_size") else (612, 792)
                    pages.append(Page(number=new_page, width=rect[0], height=rect[1]))
                except Exception:
                    pages.append(Page(number=new_page))

            text = str(getattr(item, "text", "") or "").strip()

            if isinstance(item, TitleItem):
                block = self._mk(paper_id, page_no, BlockType.TITLE, text)
            elif isinstance(item, SectionHeaderItem):
                level = int(getattr(item, "level", 1) or 1)
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                next_section_id += 1
                section_stack.append((level, next_section_id, text))
                block = self._mk(paper_id, page_no, BlockType.HEADING, text)
                block.heading_level = level
            elif isinstance(item, TableItem):
                try:
                    text = item.export_to_markdown(doc=doc).strip()
                except Exception:
                    text = ""
                if not text:
                    continue
                block = self._mk(paper_id, page_no, BlockType.TABLE, text)
            elif isinstance(item, FormulaItem):
                if not text:
                    continue
                block = self._mk(paper_id, page_no, BlockType.FORMULA, text)
            elif isinstance(item, PictureItem):
                block = self._mk(paper_id, page_no, BlockType.FIGURE, "")
                try:
                    image = item.get_image(doc)
                    if image is not None:
                        out = figure_dir / f"p{page_no:02d}_{next_order:03d}_{paper_id[-6:]}.png"
                        image.save(out, format="PNG")
                        block.image_path = str(out)
                except Exception:
                    pass
            elif isinstance(item, CodeItem):
                if not text:
                    continue
                block = self._mk(paper_id, page_no, BlockType.CODE, text)
            elif isinstance(item, TextItem):
                if not text:
                    continue
                block = self._mk(paper_id, page_no, BlockType.PARAGRAPH, text)
            else:
                continue

            block.section_path = [s[2] for s in section_stack]
            block.section_id = section_stack[-1][1] if section_stack else None
            block.metadata["section_path"] = list(block.section_path)
            block.order = next_order
            block.id = make_block_id(paper_id, page_no, next_order)
            pages[-1].blocks.append(block)
            next_order += 1

        if not pages or not pages[0].blocks:
            raise ParseError("Docling produced no extractable content.")

        paper = Paper(
            id=paper_id,
            filename=path.name,
            page_count=len(pages),
            file_sha256=_sha256_of(path),
            size_bytes=path.stat().st_size,
            source_path=str(path),
            parser=self.name,
            pages=pages,
        )
        counts: dict[str, int] = {}
        for b in paper.all_blocks():
            counts[b.block_type.value] = counts.get(b.block_type.value, 0) + 1
        paper.metadata["block_counts"] = counts
        for b in paper.pages[0].blocks:
            if b.block_type == BlockType.TITLE:
                paper.title = b.text
                break
        return paper

    @staticmethod
    def _mk(paper_id: str, page_no: int, btype: BlockType, text: str) -> Block:
        return Block(
            id="", paper_id=paper_id, block_type=btype, text=text, page=page_no, order=-1
        )


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
