import csv
from pathlib import Path

from src.loaders.base import BaseLoader, LoadedDocument, LoadedPage, LoaderError


class CSVLoader(BaseLoader):

    def load(self, path: Path) -> LoadedDocument:

        if not path.exists():
            raise LoaderError(f"File not found: {path}")

        if path.suffix.lower() != ".csv":
            raise LoaderError(
                f"Unsupported file type: {path.suffix}. Only CSV files are supported."
            )

        try:
            rows = []

            with open(path, "r", encoding="utf-8", newline="") as file:

                reader = csv.DictReader(file)

                for row in reader:
                    row_text = " | ".join(
                        f"{key}: {value}"
                        for key, value in row.items()
                    )

                    rows.append(row_text)

            text = "\n".join(rows)

            page = LoadedPage(
                text=text,
                page_number=1,
                metadata={}
            )

            return LoadedDocument(
                source_path=str(path),
                document_name=path.name,
                mime_type="text/csv",
                pages=[page],
                metadata={}
            )

        except Exception as e:
            raise LoaderError(f"Failed to load CSV: {e}")