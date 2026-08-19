from pathlib import Path

from src.loaders.pdf_loader import PDFLoader


def main():
    pdf_path = Path("data/raw/test_paper.pdf")

    print("Loading:", pdf_path)

    loader = PDFLoader()
    document = loader.load(pdf_path)

    print("\n" + "=" * 60)
    print("INGESTION TEST")
    print("=" * 60)

    print("Document:", document.document_name)
    print("Pages:", len(document.pages))

    total_figures = 0
    total_tables = 0
    total_formulas = 0
    pages_with_sections = 0

    for page in document.pages:
        text = page.text

        sections = page.metadata.get("sections", [])
        figures = page.metadata.get("figures", [])

        total_figures += len(figures)

        if sections:
            pages_with_sections += 1

        if "|---|" in text or "| --- |" in text:
            total_tables += 1

        if "$$" in text:
            total_formulas += text.count("$$") // 2

        print(
            f"Page {page.page_number}: "
            f"{len(text)} chars | "
            f"{len(sections)} sections | "
            f"{len(figures)} figures"
        )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print("Pages:", len(document.pages))
    print("Pages with sections:", pages_with_sections)
    print("Figures:", total_figures)
    print("Pages containing tables:", total_tables)
    print("Formulas:", total_formulas)


if __name__ == "__main__":
    main()