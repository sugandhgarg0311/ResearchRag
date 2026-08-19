import io

from google import genai
from google.genai import types
from PIL import Image

from src.config import GEMINI_API_KEY, VLM_MAX_TOKENS, VLM_MODEL

PROMPT = (
    "Describe this figure from a research paper for someone who cannot see it. "
    "Cover the figure type (chart, diagram, plot, photo, screenshot, ...), "
    "what it shows, axis/legend labels if present, and the key takeaway. "
    "Be concise and factual."
)


class FigureDescriber:
    """Figure -> VLM -> figure description."""

    def __init__(self, model: str = VLM_MODEL, max_tokens: int = VLM_MAX_TOKENS):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        self.max_tokens = max_tokens

    def _to_png_bytes(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()

    def describe(self, image: Image.Image) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=self._to_png_bytes(image), mime_type="image/png"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text.strip()
