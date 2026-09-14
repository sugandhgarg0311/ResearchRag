"""Retrieval and evidence models: queries, stage results, evidence packages."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from researchrag.models.chunk import Chunk


class RetrievedChunk(BaseModel):
    """A chunk with scores from each retrieval stage."""

    chunk: Chunk
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rrf_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None


class RetrievalConfig(BaseModel):
    """Explicit per-request retrieval overrides (for experiments & tuning)."""

    dense_top_n: int | None = None
    sparse_top_n: int | None = None
    rrf_k: int | None = None
    rrf_dense_weight: float | None = None
    rrf_sparse_weight: float | None = None
    use_dense: bool = True
    use_sparse: bool = True
    use_rrf: bool = True
    reranker_top_m: int | None = None
    final_top_k: int | None = None
    use_reranker: bool = True
    use_parent_context: bool | None = None


class StageTimings(BaseModel):
    dense_ms: float | None = None
    sparse_ms: float | None = None
    fusion_ms: float | None = None
    rerank_ms: float | None = None
    assembly_ms: float | None = None
    total_ms: float = 0.0


class RetrievalResult(BaseModel):
    """Full observability for one retrieval (used by UI debug view & experiments)."""

    query: str
    paper_id: str | None = None
    config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    dense_candidates: int = 0
    sparse_candidates: int = 0
    fused_candidates: int = 0
    reranked_candidates: int = 0
    final_count: int = 0
    evidence: list[RetrievedChunk] = Field(default_factory=list)
    stages: StageTimings = Field(default_factory=StageTimings)


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


class ClaimProvenance(BaseModel):
    """Where in the paper a specific claim/answer part comes from."""

    chunk_id: str
    paper_id: str
    paper_title: str | None = None
    section: str
    page: int | None = None
    page_range: str | None = None
    quote: str | None = Field(
        default=None, description="Short verbatim quote supporting the claim"
    )
    relevance: float | None = None
    source: Literal["paper"] = "paper"


class Answer(BaseModel):
    """A grounded answer with full provenance."""

    question: str
    answer: str
    mode: str = Field(
        description="'generative' (LLM) or 'extractive' (offline fallback)"
    )
    model: str | None = None
    citations: list[ClaimProvenance] = Field(default_factory=list)
    evidence: list[Chunk] = Field(default_factory=list)
    sufficiency: str = Field(
        description="'sufficient' | 'partial' | 'insufficient'"
    )
    confidence: str = Field(
        description="'explicit' | 'strongly_inferred' | 'weakly_inferred' | "
        "'external' | 'unknown'"
    )
    stages: StageTimings | None = None
    tokens_used: int | None = None


# ---------------------------------------------------------------------------
# Implementation specification
# ---------------------------------------------------------------------------


class RequirementType(str, Enum):
    ARCHITECTURE = "architecture"
    MODEL = "model"
    DATASET = "dataset"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    OPTIMIZER = "optimizer"
    HYPERPARAMETER = "hyperparameter"
    LOSS = "loss_function"
    EVALUATION = "evaluation"
    INFERENCE = "inference"
    DEPENDENCY = "dependency"
    HARDWARE = "hardware"
    ENVIRONMENT = "environment"
    REPRODUCIBILITY = "reproducibility"
    AMBIGUITY = "ambiguity"


#: What the paper actually says vs. what the system had to guess.
class SourceKind(str, Enum):
    EXPLICIT = "explicit"  # stated verbatim in the paper
    INFERRED = "inferred"  # reasonable inference from the paper
    EXTERNAL = "external"  # from outside the paper (repo, docs)
    UNKNOWN = "unknown"  # the paper does not specify this


class RequirementStatus(str, Enum):
    OPEN = "open"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


class SpecRequirement(BaseModel):
    """One traceable implementation requirement."""

    id: str = Field(description="e.g. REQ-001")
    requirement: str
    type: RequirementType
    source: SourceKind = SourceKind.EXPLICIT
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    implementation_implication: str = ""
    status: RequirementStatus = RequirementStatus.OPEN
    # Provenance (required for EXPLICIT/INFERRED; EXTERNAL uses source_ref)
    source_section: str | None = None
    source_page: int | None = None
    source_chunk_id: str | None = None
    source_quote: str | None = Field(
        default=None, description="Verbatim quote backing an explicit requirement"
    )
    source_ref: str | None = Field(
        default=None, description="External reference (URL, repo path, doc)"
    )
    notes: str = ""


class ImplementationSpec(BaseModel):
    """Structured implementation specification derived from paper evidence.

    This is the intermediate artifact between 'understanding' and 'code'.
    Every decision in generated code must trace back to a requirement id.
    """

    id: str
    paper_id: str
    title: str
    summary: str = ""
    requirements: list[SpecRequirement] = Field(default_factory=list)
    task_decomposition: list[str] = Field(
        default_factory=list, description="Ordered implementation tasks"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Recommended dependency list"
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Gaps the paper does not resolve (never invented as facts)",
    )
    generated_by: str = ""
    created_at: str = ""
    version: int = 1

    # ------------------------------------------------------------------
    def requirements_by_type(self, rtype: RequirementType) -> list[SpecRequirement]:
        return [r for r in self.requirements if r.type == rtype]

    def stats(self) -> dict[str, int]:
        by_source: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for r in self.requirements:
            by_source[r.source.value] = by_source.get(r.source.value, 0) + 1
            by_type[r.type.value] = by_type.get(r.type.value, 0) + 1
        return {
            "total": len(self.requirements),
            "by_source": by_source,
            "by_type": by_type,
        }
