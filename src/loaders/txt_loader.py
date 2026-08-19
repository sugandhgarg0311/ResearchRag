from pathlib import Path
import hashlib

from src.loaders.base import BaseLoader, LoadedDocument, LoadedPage, LoaderError


class TXTLoader(BaseLoader):

    def load(self, path: Path) -> LoadedDocument:
        if not path.exists():
            raise LoaderError(f"File not found: {path}")

        if path.suffix.lower() != ".txt":
            raise LoaderError(
                f"Unsupported file type: {path.suffix}. Only TXT files are supported."
            )

        try:
            text = path.read_text(encoding="utf-8")

            # Generate a stable ID for this document
            document_id = hashlib.sha256(
                str(path.resolve()).encode("utf-8")
            ).hexdigest()

            page = LoadedPage(
                text=text,
                page_number=1,
                metadata={}
            )

            return LoadedDocument(
                source_path=str(path),
                document_name=path.name,
                mime_type="text/plain",
                pages=[page],
                metadata={},
                document_id=document_id
            )

        except Exception as e:
            raise LoaderError(f"Failed to load TXT: {e}")