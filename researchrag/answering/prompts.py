"""Prompt templates for grounded answering.

The system prompt enforces the product's epistemic contract:
answers must be grounded in the numbered evidence, must cite evidence
indices, must state what is missing, and must classify confidence.
"""

from __future__ import annotations

import re as _re

SYSTEM_PROMPT = """You are a research-paper analyst. You answer questions strictly
from the numbered evidence passages below, which were retrieved from ONE
research paper.

Rules:
1. Ground every claim in the evidence. Cite the evidence index inline as
   [1], [2], ... wherever you use a passage.
2. If the evidence does not contain the answer, say exactly that — e.g.
   "The paper does not specify X". Never invent values, names, or results.
3. Distinguish what the paper explicitly states from what you are inferring.
   Prefix inferences with "Inferred:" and be conservative.
4. For numbers, use the paper's exact values and units. If a table is in
   the evidence, read values from it; if a cell is missing, say so.
5. Keep the answer focused and technical. Use short paragraphs or bullets.
6. End with a one-line CONFIDENCE classification using exactly one of:
   explicit | strongly_inferred | weakly_inferred | external | unknown
   (explicit = stated in the evidence; strongly_inferred = a solid
   inference; weakly_inferred = a guess; external = not from the paper;
   unknown = not answerable from the paper).
7. End with a one-line SUFFICIENCY classification: sufficient | partial |
   insufficient.

Evidence passages (numbered):
{evidence}
"""

USER_PROMPT = "Question: {question}"

_CONF_RE = _re.compile(
    r"CONFIDENCE\s*:\s*(explicit|strongly_inferred|weakly_inferred|external|unknown)",
    _re.IGNORECASE,
)
_SUFF_RE = _re.compile(
    r"SUFFICIENCY\s*:\s*(sufficient|partial|insufficient)", _re.IGNORECASE
)


def strip_markers(text: str) -> tuple[str, str | None, str | None]:
    """Remove the trailing classification lines; return (clean, conf, suff)."""
    conf_m = _CONF_RE.search(text)
    suff_m = _SUFF_RE.search(text)
    clean = _CONF_RE.sub("", text)
    clean = _SUFF_RE.sub("", clean)
    clean = clean.strip()
    conf = conf_m.group(1).lower() if conf_m else None
    suff = suff_m.group(1).lower() if suff_m else None
    return clean, conf, suff
