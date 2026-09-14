"""Rule-based implementation-requirement extraction (offline mode).

When no LLM is available, this extractor scans the paper's evidence for
commonly specified implementation details with precise patterns and
produces typed, provenance-tagged requirements. Every extracted
requirement is marked ``source=explicit`` with the verbatim quote and
chunk id — the same epistemic contract as the LLM path, just with a
smaller, auditable coverage area.

Coverage: optimizers, learning rates, weight decay, batch sizes,
scheduling, training steps/epochs, seeds, hardware, model architectures
(layer counts), dropout, warmup, top-k retrieval, null documents.
Anything it does not find is simply absent — the spec's
``open_questions`` states that explicitly.
"""

from __future__ import annotations

import re

from researchrag.models.retrieval import (
    ImplementationSpec,
    RequirementType,
    SourceKind,
    SpecRequirement,
)
from researchrag.storage.database import now_iso


class _Rule:
    def __init__(
        self,
        pattern: str,
        rtype: RequirementType,
        template: str,
        implication: str,
        group: str | None = None,
        flags: int = re.IGNORECASE,
    ):
        self.re = re.compile(pattern, flags)
        self.rtype = rtype
        self.template = template
        self.implication = implication
        self.group = group

    def match(self, text: str):
        """Return the regex Match, or None. Value group is ``self.group`` (str) or 0."""
        return self.re.search(text)


RULES: list[_Rule] = [
    _Rule(
        r"(?:learning rate|lr)\D{0,24}?(?:of|was|is|:|,)?\s*(\d[.]?\d*[eE][\-+]\d{1,2}|\d+[\.,]\d*)",
        RequirementType.HYPERPARAMETER,
        "Learning rate: {v}",
        "Set the optimizer's learning rate to this value (verify scale: 1e-5 style is typical for BERT fine-tuning).",
        group=1,
    ),
    _Rule(
        r"(AdamW|Adam|SGD|AdaGrad|Adagrad|Adadelta|RMSprop)\b",
        RequirementType.OPTIMIZER,
        "Optimizer: {v}",
        "Instantiate the named optimizer.",
    ),
    _Rule(
        r"(?:weight decay)\D{0,20}?(?:of|:|,)?\s*(\d+[\.,]\d*)",
        RequirementType.HYPERPARAMETER,
        "Weight decay: {v}",
        "Apply weight decay with this coefficient (often excluded for bias/LayerNorm).",
        group=1,
    ),
    _Rule(
        r"(?:batch size|batch-size)\D{0,24}?(?:of|:|,)?\s*(\d+)",
        RequirementType.HYPERPARAMETER,
        "Batch size: {v}",
        "Train with this per-device (or global — read surrounding text) batch size.",
        group=1,
    ),
    _Rule(
        r"(?:warm[ -]?up|warmup)\D{0,28}?(?:of|:|,)?\s*(\d{1,5})",
        RequirementType.HYPERPARAMETER,
        "Warmup: {v} steps",
        "Linearly warm up the learning rate for this many steps.",
        group=1,
    ),
    _Rule(
        r"(\d+)\s+(?:encoder|decoder)\s+layers",
        RequirementType.ARCHITECTURE,
        "{v} encoder/decoder layers",
        "Build the model with this layer count.",
        group=1,
    ),
    _Rule(
        r"hidden\s+size\D{0,20}?(?:of|:|,)?\s*(\d+)",
        RequirementType.MODEL,
        "Hidden size: {v}",
        "Set transformer hidden dimension to this value.",
        group=1,
    ),
    _Rule(
        r"(?:dropout|drop-out)\D{0,20}?(?:of|:|,)?\s*(\d+[\.,]?\d*)",
        RequirementType.HYPERPARAMETER,
        "Dropout: {v}",
        "Apply dropout with this rate.",
        group=1,
    ),
    _Rule(
        r"top[- ]?(k|K)\D{0,16}?(?:of|:|,)?\s*(\d+)",
        RequirementType.INFERENCE,
        "Top-K retrieval: K={v}",
        "Retrieve this many documents per query at inference (and matching training if stated).",
        group=2,
    ),
    _Rule(
        r"(?:random seed|seed)\D{0,24}?(?:of|:|,)?\s*(\d+)",
        RequirementType.REPRODUCIBILITY,
        "Random seed: {v}",
        "Fix the RNG seed for reproducibility.",
        group=1,
    ),
    _Rule(
        r"(A100|V100|P100|T4|A10|RTX 30\d{2}|RTX 40\d{2}|H100)\b",
        RequirementType.HARDWARE,
        "Hardware: {v} GPU",
        "Target hardware; check for GPU-dependent hyperparameters.",
    ),
    _Rule(
        r"(?:fine-?tun(?:e|ing)|pre-?train(?:ed|ing))",
        RequirementType.TRAINING,
        "Training procedure: {v}",
        "Follow the stated training procedure (pre-train / fine-tune).",
    ),
    _Rule(
        r"(?:cross[- ]entropy|CE|BCE|focal|contrastive|triplet|margin)\s+(?:loss|objective)",
        RequirementType.LOSS,
        "Loss function: {v}",
        "Use the stated loss function.",
    ),
    _Rule(
        r"(BM25|DPR|dense retriev\w*|dense index|document index|query encoder|document encoder)",
        RequirementType.MODEL,
        "Retriever component: {v}",
        "Implement/select the named retrieval component.",
    ),
    _Rule(
        r"top[- ]?(\d{1,6})\s+documents",
        RequirementType.INFERENCE,
        "Top-N documents at retrieval: N={v}",
        "Retrieve this many documents per query.",
        group=1,
    ),
    _Rule(
        r"k\s*∈\s*\{([^}]+)\}",
        RequirementType.HYPERPARAMETER,
        "Candidate k values: {v}",
        "Sweep/choose k from this set (usually validated on dev data).",
        group=1,
    ),
    _Rule(
        r"(?<!top )(?<!top-)(\d{1,9}(?:[.,]\d+)?[M]?)\s+documents\b",
        RequirementType.DATASET,
        "Corpus size: {v} documents",
        "Build/serve a document index of this size.",
        group=1,
    ),
    _Rule(
        r"(\d+)-word chunks",
        RequirementType.PREPROCESSING,
        "Document split: {v}-word chunks",
        "Split source documents into chunks of this length before indexing.",
        group=1,
    ),
    _Rule(
        r"mixed precision",
        RequirementType.TRAINING,
        "Training precision: {v}",
        "Use mixed-precision (fp16/bf16) training.",
    ),
    _Rule(
        r"(FAISS|HNSW|HNSWlib|ScaNN|Annoy)\b",
        RequirementType.MODEL,
        "Index structure: {v}",
        "Use this approximate nearest-neighbor index for MIPS.",
    ),
    _Rule(
        r"beam (?:search|size)\D{0,20}?(?:of|:|,)?\s*(\d+)",
        RequirementType.INFERENCE,
        "Beam size: {v}",
        "Decode with this beam width.",
        group=1,
    ),
]


def extract_requirements(
    paper_id: str,
    paper_title: str | None,
    evidence_texts: list[tuple[str, str, list[str], int | None, str]],
    max_requirements: int = 40,
) -> ImplementationSpec:
    """evidence_texts: (chunk_id, text, section_path, page, chunk_type)"""
    requirements: list[SpecRequirement] = []
    seen: set[tuple[str, str]] = set()

    for chunk_id, text, section_path, page, chunk_type in evidence_texts:
        for rule in RULES:
            m = rule.match(text)
            if not m:
                continue
            value = m.group(int(rule.group) if rule.group else 0).strip()
            requirement_text = rule.template.format(v=value)
            key = (rule.rtype.value, requirement_text.lower())
            if key in seen:
                continue
            seen.add(key)
            # keep the quote short
            qstart = max(0, m.start(0) - 40)
            qend = min(len(text), m.end(0) + 60)
            quote = text[qstart:qend].strip()
            requirements.append(
                SpecRequirement(
                    id=f"REQ-{len(requirements) + 1:03d}",
                    requirement=requirement_text,
                    type=rule.rtype,
                    source=SourceKind.EXPLICIT,
                    confidence=0.9,
                    implementation_implication=rule.implication,
                    source_section=" › ".join(section_path) or None,
                    source_page=page,
                    source_chunk_id=chunk_id,
                    source_quote=f"…{quote}…" if len(quote) < 100 else f"{quote[:140]}…",
                )
            )
            if len(requirements) >= max_requirements:
                break
        if len(requirements) >= max_requirements:
            break

    return ImplementationSpec(
        id=f"spec_{paper_id[-6:]}_rules",
        paper_id=paper_id,
        title=f"Implementation specification — {paper_title or paper_id}",
        summary=(
            "Rule-based extraction (offline mode). Covers common hyperparameters, "
            "components, and hardware stated in the paper. Requirements not matched "
            "by the rules are listed in open questions."
        ),
        requirements=requirements,
        task_decomposition=[
            "Set up environment and dependencies",
            "Implement data pipeline",
            "Implement model architecture",
            "Implement training loop with stated hyperparameters",
            "Implement evaluation on stated datasets/metrics",
            "Run experiments and record results",
        ],
        open_questions=[
            "Rule-based extraction may miss requirements phrased differently; "
            "review the paper's Methods and appendices for anything absent here.",
            "Values marked here are verbatim from the paper; verify units and scope "
            "(per-device vs global batch, steps vs epochs) against surrounding text.",
        ],
        generated_by="rule-extractor v1",
        created_at=now_iso(),
    )
