"""Central configuration for Research RAG.

All tunables live here, loaded from environment variables / ``.env``.
See ``.env.example`` for documentation of every variable.

Design rules:
  * No secrets may be logged. ``Settings`` implements ``safe_repr``.
  * Every retrieval/generation parameter is explicit so experiments can
    override them via the API instead of editing code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (researchrag/config.py -> parent.parent)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RESEARCHRAG_",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_dir: Path = Field(default=DATA_ROOT, description="Root for all persistent data")
    database_path: Path | None = Field(
        default=None, description="SQLite path (default: <data_dir>/researchrag.db)"
    )
    qdrant_path: Path | None = Field(
        default=None, description="Qdrant local storage (default: <data_dir>/qdrant)"
    )
    artifacts_dir: Path | None = Field(
        default=None, description="Generated artifacts (default: <data_dir>/artifacts)"
    )
    frontend_dir: Path = Field(
        default=REPO_ROOT / "frontend" / "dist",
        description="Built frontend to serve statically (optional)",
    )
    evals_dir: Path | None = Field(
        default=None,
        description="Eval datasets directory (default: <repo>/evals if present, else <data_dir>/evals)",
    )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    parser_backend: str = Field(
        default="pymupdf",
        description="PDF parser backend: 'pymupdf' (default, lightweight) or 'docling' (install extra)",
    )
    figure_max_pixels: int = Field(
        default=1_600_000, description="Downscale figures larger than this many pixels"
    )
    ocr_language: str = Field(default="eng", description="OCR language (if OCR enabled)")
    enable_ocr: bool = Field(
        default=False,
        description="Run OCR on pages with little extractable text (requires tesseract binary)",
    )

    # ------------------------------------------------------------------
    # Chunking (hierarchical: parent context + child retrieval chunks)
    # ------------------------------------------------------------------
    chunk_strategy: str = Field(
        default="section_aware", description="'section_aware' or 'fixed_size'"
    )
    chunk_max_tokens: int = Field(
        default=400, description="Max tokens per child retrieval chunk"
    )
    chunk_parent_max_tokens: int = Field(
        default=3000,
        description="Max tokens retained for a parent (section-level) context chunk",
    )
    chunk_overlap_tokens: int = Field(
        default=0,
        description="Token overlap between child chunks of a large section (0 = none; "
        "section-aware chunking is boundary-driven, not window-based)",
    )
    fixed_chunk_tokens: int = Field(default=500, description="Fixed-size fallback chunk size")
    fixed_chunk_overlap_tokens: int = Field(
        default=80, description="Fixed-size fallback overlap"
    )

    # ------------------------------------------------------------------
    # Embeddings (dense)
    # ------------------------------------------------------------------
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="fastembed model name for dense embeddings (768-d, ONNX runtime)",
    )
    embedding_batch_size: int = Field(default=32, description="Embedding batch size")
    # bge models are trained asymmetrically: prepend this instruction to queries.
    embedding_query_instruction: str = Field(
        default="Given a research question about a paper, retrieve relevant passages that "
        "answer it.",
        description="Instruction prepended to queries for asymmetric (bge) models",
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    dense_top_n: int = Field(default=50, description="Candidates from dense retrieval")
    sparse_top_n: int = Field(default=50, description="Candidates from sparse (BM25) retrieval")
    rrf_k: int = Field(
        default=60,
        description="RRF constant k (Cormack et al. 2009; 60 is the published default)",
    )
    rrf_dense_weight: float = Field(default=1.0, description="Weight of the dense ranking in RRF")
    rrf_sparse_weight: float = Field(default=1.0, description="Weight of the sparse ranking in RRF")
    reranker_top_m: int = Field(
        default=30, description="Fused candidates sent to the reranker"
    )
    final_top_k: int = Field(default=8, description="Final evidence chunks returned")
    evidence_max_tokens: int = Field(
        default=4000,
        description="Global token budget for the assembled evidence package",
    )
    use_parent_context: bool = Field(
        default=True,
        description="Expand retrieved child chunks with their parent section context",
    )

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    reranker: str = Field(
        default="local",
        description="'local' (ONNX cross-encoder via fastembed), 'cohere', 'jina', or 'none'",
    )
    reranker_model: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        description="ONNX cross-encoder model for the local reranker",
    )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_provider: str = Field(
        default="openai_compatible",
        description="'openai_compatible', 'anthropic', or 'offline'",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name for the active provider (or Ollama/vLLM model id)",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible endpoint (e.g. http://localhost:11434/v1 for Ollama). "
        "Default: provider's public API.",
    )
    llm_api_key: str | None = Field(default=None, description="API key for the LLM provider")
    llm_temperature: float = Field(default=0.2, description="Sampling temperature")
    llm_max_tokens: int = Field(default=4096, description="Max output tokens")
    llm_timeout_s: float = Field(default=180.0, description="Per-request LLM timeout (s)")
    llm_max_retries: int = Field(
        default=2, description="Retries on transient (5xx / timeout) failures"
    )

    # ------------------------------------------------------------------
    # Vision (optional figure understanding)
    # ------------------------------------------------------------------
    vlm_model: str | None = Field(
        default=None,
        description="Vision model for figure description (e.g. gpt-4o-mini). Requires an "
        "LLM API key. When unset, figures are indexed by caption + surrounding text only.",
    )

    # ------------------------------------------------------------------
    # Execution sandbox
    # ------------------------------------------------------------------
    execution_timeout_s: int = Field(
        default=120, description="Wall-clock timeout for generated-code execution"
    )
    execution_memory_mb: int = Field(
        default=1024, description="RAM limit for sandboxed execution (Linux rlimit)"
    )
    execution_allow_network: bool = Field(
        default=False,
        description="Allow network access inside the execution sandbox (default: blocked)",
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Verbose/debug mode (exposes stack traces)")

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.database_path or (self.data_dir / "researchrag.db")

    @property
    def qdrant_dir(self) -> Path:
        return self.qdrant_path or (self.data_dir / "qdrant")

    @property
    def artifacts_root(self) -> Path:
        return self.artifacts_dir or (self.data_dir / "artifacts")

    @property
    def evals_root(self) -> Path:
        if self.evals_dir:
            return Path(self.evals_dir)
        repo_evals = REPO_ROOT / "evals"
        if repo_evals.exists():
            return repo_evals
        return self.data_dir / "evals"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.qdrant_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def masked_dict(settings: Settings) -> dict:
    """Settings as a dict with secrets redacted (for JSON/API)."""
    d = settings.model_dump()
    for key in list(d):
        if "key" in key.lower() or "token" in key.lower():
            if d.get(key):
                d[key] = "***"
    return d


def safe_repr(settings: Settings) -> str:
    """Log-safe representation with secrets redacted."""
    return str(masked_dict(settings))
