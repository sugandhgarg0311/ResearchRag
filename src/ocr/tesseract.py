import fitz
import pytesseract
from PIL import Image


class TesseractOCR:

    def __init__(self, dpi=300, lang="eng"):
        self.zoom = dpi / 72
        self.lang = lang

    def _render(self, page):
        pix = page.get_pixmap(
            matrix=fitz.Matrix(self.zoom, self.zoom),
            colorspace=fitz.csGRAY,
            alpha=False,
        )

        return Image.frombytes(
            "L",
            (pix.width, pix.height),
            pix.samples
        )

    def extract_text(self, pdf_path, page_number):
        with fitz.open(pdf_path) as doc:
           page = doc[page_number]
           image = self._render(page)
        return pytesseract.image_to_string(image, lang=self.lang)