from pathlib import Path

from src.loaders.base import BaseLoader, LoadedDocument, LoadedPage, LoaderError


class MDLoader(BaseLoader):

    def load(self, path: Path) -> LoadedDocument:

        if not path.exists():
            raise LoaderError(f"File not found: {path}")

        if path.suffix.lower() != ".md":
            raise LoaderError(
                f"Unsupported file type: {path.suffix}. Only Markdown files are supported."
            )

        try:
            text = path.read_text(encoding="utf-8")

            page = LoadedPage(
                text=text,
                page_number=1,
                metadata={}
            )

            return LoadedDocument(
                source_path=str(path),
                document_name=path.name,
                mime_type="text/markdown",
                pages=[page],
                metadata={}
            )

        except Exception as e:
            raise LoaderError(f"Failed to load Markdown: {e}")