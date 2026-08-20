from pathlib import Path
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)


PDF_PATH = Path("data/raw/test_paper.pdf")


def main():

    print("\n" + "=" * 60)
    print("DOCLING INTERNAL PROFILER")
    print("=" * 60)

    pipeline_options = PdfPipelineOptions()

    # Our current experiment setting
    pipeline_options.do_ocr = False

    # Keep table processing enabled for ResearchRAG
    pipeline_options.do_table_structure = True
    pipeline_options.force_backend_text = True

    # Already disabled in your loader
    pipeline_options.do_formula_enrichment = False
    pipeline_options.generate_picture_images = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    # IMPORTANT:
    # Enable Docling's own pipeline profiling.
    settings.debug.profile_pipeline_timings = True

    print("\nStarting conversion...")

    result = converter.convert(PDF_PATH)

    print("\nConversion complete.")

    print("\n" + "=" * 60)
    print("PIPELINE TIMINGS")
    print("=" * 60)

    print(result.timings)

    print("\n" + "=" * 60)
    print("PIPELINE TOTAL")
    print("=" * 60)

    print(
        result.timings["pipeline_total"].times
    )


if __name__ == "__main__":
    main()