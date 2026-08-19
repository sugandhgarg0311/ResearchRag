from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from docling_core.types.doc import (
    CodeItem,
    FormulaItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from src.loaders.base import (
    BaseLoader,
    DocumentBlock,
    LoadedDocument,
    LoadedPage,
    LoaderError,
)


class PDFLoader(BaseLoader):

    def __init__(self):

        pipeline_options = PdfPipelineOptions()

        # We only need normal PDF text/structure for V1.
        pipeline_options.do_formula_enrichment = False
        pipeline_options.generate_picture_images = False

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def load(self, path: Path) -> LoadedDocument:

        path = Path(path)

        if not path.exists():
            raise LoaderError(f"File not found: {path}")

        if not path.is_file():
            raise LoaderError(f"Path is not a file: {path}")

        if path.suffix.lower() != ".pdf":
            raise LoaderError(
                "Only PDF files are supported."
            )

        # --------------------------------------------------
        # Parse PDF
        # --------------------------------------------------

        try:
            result = self._converter.convert(path)
            doc = result.document

        except Exception as e:
            raise LoaderError(
                f"Failed to parse PDF: {e}"
            ) from e

        # --------------------------------------------------
        # Extract blocks
        # --------------------------------------------------

        blocks = self._extract_blocks(doc)

        if not blocks:
            raise LoaderError(
                "No extractable content found."
            )

        # --------------------------------------------------
        # Group blocks by page
        # --------------------------------------------------

        page_groups = defaultdict(list)

        for block in blocks:
            page_groups[block.page_number].append(block)

        pages = [
            LoadedPage(
                page_number=page_number,
                blocks=page_groups[page_number],
            )
            for page_number in sorted(page_groups)
        ]

        # --------------------------------------------------
        # Stable document ID
        # --------------------------------------------------

        document_id = self._calculate_document_id(path)

        # --------------------------------------------------
        # Document metadata
        # --------------------------------------------------

        metadata = {
            "loader": "docling",
            "page_count": len(pages),
            "block_count": len(blocks),
            "block_types": self._count_block_types(blocks),
        }

        return LoadedDocument(
            source_path=str(path),
            document_name=path.name,
            mime_type="application/pdf",
            pages=pages,
            metadata=metadata,
            document_id=document_id,
        )

    # ======================================================
    # BLOCK EXTRACTION
    # ======================================================

    def _extract_blocks(self, doc) -> list[DocumentBlock]:

        blocks = []

        section_stack = []

        section_id = 0
        order = 0

        for item, docling_level in doc.iterate_items():

            # --------------------------------------------------
            # Page number
            # --------------------------------------------------

            prov = getattr(item, "prov", None)

            if not prov:
                continue

            page_number = getattr(
                prov[0],
                "page_no",
                None,
            )

            if page_number is None:
                continue

            page_number = int(page_number)

            # ==================================================
            # TITLE
            # ==================================================

            if isinstance(item, TitleItem):

                text = self._get_text(item)

                if not text:
                    continue

                blocks.append(
                    DocumentBlock(
                        block_type="title",
                        text=text,
                        page_number=page_number,
                        order=order,
                        metadata={
                            "section_path": [],
                        },
                    )
                )

                order += 1
                continue

            # ==================================================
            # HEADING
            # ==================================================

            if isinstance(item, SectionHeaderItem):

                text = self._get_text(item)

                if not text:
                    continue

                level = self._infer_heading_level(text)

                if level is None:
                    level = getattr(
                        item,
                        "level",
                        None,
                    )

                if not isinstance(level, int):
                    level = 1

                # Remove current/deeper sections.
                while (
                    section_stack
                    and section_stack[-1][0] >= level
                ):
                    section_stack.pop()

                section_id += 1

                section_stack.append(
                    (
                        level,
                        section_id,
                        text,
                    )
                )

                section_path = [
                    section[2]
                    for section in section_stack
                ]

                blocks.append(
                    DocumentBlock(
                        block_type="heading",
                        text=text,
                        page_number=page_number,
                        order=order,
                        metadata={
                            "section_id": section_id,
                            "section_path": section_path,
                            "heading_level": level,
                        },
                    )
                )

                order += 1
                continue

            # --------------------------------------------------
            # Current section context
            # --------------------------------------------------

            current_section_id = (
                section_stack[-1][1]
                if section_stack
                else None
            )

            section_path = [
                section[2]
                for section in section_stack
            ]

            # ==================================================
            # TABLE
            # ==================================================

            if isinstance(item, TableItem):

                try:
                    text = item.export_to_markdown(
                        doc=doc
                    )
                except Exception:
                    text = str(item)

                text = text.strip()

                if not text:
                    continue

                blocks.append(
                    DocumentBlock(
                        block_type="table",
                        text=text,
                        page_number=page_number,
                        order=order,
                        metadata={
                            "section_id": current_section_id,
                            "section_path": section_path,
                            "atomic": True,
                        },
                    )
                )

                order += 1
                continue

            # ==================================================
            # FORMULA
            # ==================================================

            if isinstance(item, FormulaItem):

                text = self._get_text(item)

                if not text:
                    continue

                blocks.append(
                    DocumentBlock(
                        block_type="formula",
                        text=text,
                        page_number=page_number,
                        order=order,
                        metadata={
                            "section_id": current_section_id,
                            "section_path": section_path,
                            "atomic": True,
                        },
                    )
                )

                order += 1
                continue

            # ==================================================
            # FIGURE
            # ==================================================

            if isinstance(item, PictureItem):

                caption = self._extract_caption(
                    item,
                    doc,
                )

                if caption:
                    text = caption
                else:
                    text = "[Figure]"

                blocks.append(
                    DocumentBlock(
                        block_type="figure",
                        text=text,
                        page_number=page_number,
                        order=order,
                        metadata={
                            "section_id": current_section_id,
                            "section_path": section_path,
                            "caption": caption,
                            "atomic": True,
                        },
                    )
                )

                order += 1
                continue

            # ==================================================
            # CODE
            # ==================================================

            if isinstance(item, CodeItem):

                text = self._get_text(item)

                if not text:
                    continue

                blocks.append(
                    DocumentBlock(
                        block_type="code",
                        text=text,
                        page_number=page_number,
                        order=order,
                        metadata={
                            "section_id": current_section_id,
                            "section_path": section_path,
                            "atomic": True,
                        },
                    )
                )

                order += 1
                continue

            # ==================================================
            # NORMAL TEXT
            # ==================================================

            if isinstance(item, TextItem):

                text = self._get_text(item)

                if not text:
                    continue

                item_type = type(item).__name__

                if "list" in item_type.lower():
                    block_type = "list_item"
                else:
                    block_type = "text"

                blocks.append(
                    DocumentBlock(
                        block_type=block_type,
                        text=text,
                        page_number=page_number,
                        order=order,
                        metadata={
                            "section_id": current_section_id,
                            "section_path": section_path,
                        },
                    )
                )

                order += 1

        return blocks

    # ======================================================
    # FIGURE CAPTION
    # ======================================================

    @staticmethod
    def _extract_caption(item, doc) -> str:

        caption_method = getattr(
            item,
            "caption_text",
            None,
        )

        if not callable(caption_method):
            return ""

        try:
            caption = caption_method(doc=doc)

            if caption:
                return str(caption).strip()

        except Exception:
            pass

        return ""

    # ======================================================
    # HEADING LEVEL
    # ======================================================

    @staticmethod
    def _infer_heading_level(text: str) -> int | None:

        match = re.match(
            r"^\s*(\d+(?:\.\d+)*)\b",
            text,
        )

        if not match:
            return None

        number = match.group(1)

        return number.count(".") + 1

    # ======================================================
    # TEXT
    # ======================================================

    @staticmethod
    def _get_text(item) -> str:

        text = getattr(
            item,
            "text",
            None,
        )

        return str(text).strip() if text else ""

    # ======================================================
    # DOCUMENT ID
    # ======================================================

    @staticmethod
    def _calculate_document_id(path: Path) -> str:

        sha256 = hashlib.sha256()

        with path.open("rb") as file:

            while True:

                data = file.read(1024 * 1024)

                if not data:
                    break

                sha256.update(data)

        return sha256.hexdigest()

    # ======================================================
    # BLOCK COUNTS
    # ======================================================

    @staticmethod
    def _count_block_types(
        blocks: list[DocumentBlock],
    ) -> dict[str, int]:

        counts = {}

        for block in blocks:

            counts[block.block_type] = (
                counts.get(block.block_type, 0) + 1
            )

        return counts