from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_core.types.doc import (
    FormulaItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from src.loaders.base import (
    BaseLoader,
    LoadedDocument,
    LoadedPage,
    DocumentBlock,
    LoaderError,
)

from src.vlm.figure_describer import FigureDescriber


class PDFLoader(BaseLoader):

    """
    PDF loader using Docling.

    The loader's responsibility is to PRESERVE document structure.

    It does NOT decide how content should be chunked.

    It extracts:

        heading
        text
        table
        figure
        formula

    and stores them as DocumentBlock objects.
    """

    def __init__(
        self,
        figure_describer: FigureDescriber | None = None
    ):

        pipeline_options = PdfPipelineOptions()

        # Enable formula extraction/enrichment
        pipeline_options.do_formula_enrichment = True

        # Generate images for figures
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        self._figure_describer = (
            figure_describer
            or FigureDescriber()
        )

    # ============================================================
    # LOAD
    # ============================================================

    def load(self, path: Path) -> LoadedDocument:

        # --------------------------------------------------------
        # Validate file
        # --------------------------------------------------------

        if not path.exists():
            raise LoaderError(
                f"File not found: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise LoaderError(
                f"Unsupported file type: {path.suffix}. "
                f"Only PDF files are supported."
            )

        # --------------------------------------------------------
        # Parse PDF with Docling
        # --------------------------------------------------------

        try:

            result = self._converter.convert(path)

            doc = result.document

        except Exception as e:

            raise LoaderError(
                f"Failed to parse PDF with Docling: {e}"
            )

        # --------------------------------------------------------
        # Data structures
        # --------------------------------------------------------

        page_blocks: dict[
            int,
            list[DocumentBlock]
        ] = {}

        page_sections: dict[
            int,
            list[str]
        ] = {}

        page_figures: dict[
            int,
            list[str]
        ] = {}

        # Current section hierarchy
        section_path: list[str] = []

        # --------------------------------------------------------
        # Walk through Docling elements
        # --------------------------------------------------------

        try:

            for item, _level in doc.iterate_items():

                # Some Docling elements may not have provenance
                if not getattr(item, "prov", None):
                    continue

                page_no = item.prov[0].page_no

                # Initialize page containers
                page_blocks.setdefault(
                    page_no,
                    []
                )

                page_sections.setdefault(
                    page_no,
                    []
                )

                page_figures.setdefault(
                    page_no,
                    []
                )

                # =================================================
                # HEADING
                # =================================================

                if isinstance(
                    item,
                    (SectionHeaderItem, TitleItem)
                ):

                    level = getattr(
                        item,
                        "level",
                        1
                    )

                    # Update section hierarchy
                    section_path = (
                        section_path[:level - 1]
                        + [item.text]
                    )

                    page_blocks[page_no].append(
                        DocumentBlock(
                            block_type="heading",

                            text=item.text,

                            page_number=page_no,

                            metadata={
                                "level": level,

                                "section_path":
                                    list(section_path),
                            },
                        )
                    )

                    page_sections[page_no].append(
                        " > ".join(section_path)
                    )

                # =================================================
                # TABLE
                # =================================================

                elif isinstance(
                    item,
                    TableItem
                ):

                    table_text = (
                        item.export_to_markdown(
                            doc=doc
                        )
                    )

                    page_blocks[page_no].append(
                        DocumentBlock(
                            block_type="table",

                            text=table_text,

                            page_number=page_no,

                            metadata={
                                "section_path":
                                    list(section_path),
                            },
                        )
                    )

                # =================================================
                # FIGURE
                # =================================================

                elif isinstance(
                    item,
                    PictureItem
                ):

                    try:

                        image = item.get_image(
                            doc
                        )

                        if image is not None:

                            description = (
                                self
                                ._figure_describer
                                .describe(image)
                            )

                        else:

                            description = (
                                "[Figure could not be rendered]"
                            )

                    except Exception as e:

                        description = (
                            "[Figure description "
                            f"unavailable: {e}]"
                        )

                    page_blocks[page_no].append(
                        DocumentBlock(
                            block_type="figure",

                            text=description,

                            page_number=page_no,

                            metadata={
                                "section_path":
                                    list(section_path),

                                "description_generated_by":
                                    "vlm",
                            },
                        )
                    )

                    page_figures[page_no].append(
                        description
                    )

                # =================================================
                # FORMULA
                # =================================================

                elif isinstance(
                    item,
                    FormulaItem
                ):

                    formula = item.text.strip()

                    if formula:

                        page_blocks[page_no].append(
                            DocumentBlock(
                                block_type="formula",

                                text=formula,

                                page_number=page_no,

                                metadata={
                                    "section_path":
                                        list(section_path),
                                },
                            )
                        )

                # =================================================
                # NORMAL TEXT
                # =================================================

                elif isinstance(
                    item,
                    TextItem
                ):

                    text = item.text.strip()

                    if text:

                        page_blocks[page_no].append(
                            DocumentBlock(
                                block_type="text",

                                text=text,

                                page_number=page_no,

                                metadata={
                                    "section_path":
                                        list(section_path),
                                },
                            )
                        )

        except Exception as e:

            raise LoaderError(
                "Failed to walk Docling "
                f"document structure: {e}"
            )

        # =========================================================
        # CREATE LOADED PAGES
        # =========================================================

        pages = []

        for page_no in sorted(page_blocks):

            pages.append(
                LoadedPage(

                    blocks=page_blocks[page_no],

                    page_number=page_no,

                    metadata={
                        "sections":
                            page_sections[page_no],

                        "figures":
                            page_figures[page_no],
                    },
                )
            )

        # =========================================================
        # VALIDATION
        # =========================================================

        if not pages:

            raise LoaderError(
                "No extractable content found in PDF."
            )

        # =========================================================
        # RETURN DOCUMENT
        # =========================================================

        return LoadedDocument(

            source_path=str(path),

            document_name=path.name,

            mime_type="application/pdf",

            pages=pages,

            metadata={},
        )