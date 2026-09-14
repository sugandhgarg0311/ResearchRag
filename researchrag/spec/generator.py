"""Implementation specification generation.

Two backends behind one interface:

* **LLM** (default when a provider is configured): reads the paper's
  method/training/experiment sections as evidence and emits a structured
  JSON spec. The prompt enforces the explicit/inferred/external/unknown
  contract — every requirement must carry provenance, and gaps must be
  reported as open questions, never invented.
* **rule-extractor** (offline fallback): deterministic pattern scanning,
  every hit tagged with the verbatim quote.

Output is the same ``ImplementationSpec`` either way.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from researchrag.config import Settings
from researchrag.llm.base import BaseLLM, LLMError, LLMMessage
from researchrag.models.retrieval import (
    ImplementationSpec,
    RequirementStatus,
    RequirementType,
    SourceKind,
    SpecRequirement,
)
from researchrag.spec.extractor_rules import extract_requirements
from researchrag.storage.database import new_id, now_iso

log = logging.getLogger(__name__)

SPEC_PROMPT = """You are a senior ML engineer preparing to reimplement the method
of a research paper from scratch. Using ONLY the evidence passages below
(numbered, with section and page), produce a JSON implementation
specification.

Output EXACTLY one JSON object (no markdown fences) with this shape:
{
  "summary": "2-4 sentence summary of the method to implement",
  "requirements": [
    {
      "requirement": "one concrete, checkable requirement",
      "type": one of [architecture, model, dataset, preprocessing, training,
                      optimizer, hyperparameter, loss_function, evaluation,
                      inference, dependency, hardware, environment,
                      reproducibility, ambiguity],
      "source": one of [explicit, inferred, external, unknown],
      "confidence": number 0..1,
      "implementation_implication": "what the implementer must build/configure",
      "source_section": "section path from the evidence (or null)",
      "source_page": number or null,
      "evidence": [1, 3],
      "source_quote": "short verbatim quote when source=explicit, else null",
      "source_ref": "URL/repo when source=external, else null",
      "notes": ""
    }
  ],
  "task_decomposition": ["ordered implementation tasks"],
  "dependencies": ["python package ==version when stated, else package"],
  "open_questions": ["things the paper does NOT specify that an implementer
                     must decide"]
}

Strict rules:
1. "explicit" requirements MUST quote the paper verbatim in source_quote.
2. Anything you guess or standardize is "inferred" with lower confidence —
   NEVER mark an invention as explicit.
3. If a detail is missing from the evidence, put it in open_questions,
   optionally as an "ambiguity" requirement. Do not invent values.
4. Prefer the paper's exact numbers (learning rate, batch size, K, layers,
   dataset splits, metrics).
5. Cover: architecture, model components, datasets + splits, preprocessing,
   training loop (optimizer, lr, schedule, batch, steps, hardware),
   evaluation metrics, inference setup, reproducibility (seeds).
6. evidence must reference the numbered passages that support the item.

Evidence passages:
{evidence}
"""


class SpecGenerator:
    def __init__(self, settings: Settings, llm: BaseLLM | None):
        self.settings = settings
        self.llm = llm

    # ------------------------------------------------------------------
    def generate(
        self,
        paper_id: str,
        paper_title: str | None,
        evidence_texts: list[tuple[str, str, list[str], int | None, str]],
    ) -> ImplementationSpec:
        if self.llm is not None:
            try:
                return self._generate_llm(paper_id, paper_title, evidence_texts)
            except (LLMError, ValidationError, ValueError) as e:
                log.warning("LLM spec generation failed (%s); rule fallback", e)
        return extract_requirements(paper_id, paper_title, evidence_texts)

    # ------------------------------------------------------------------
    def _generate_llm(
        self,
        paper_id: str,
        paper_title: str | None,
        evidence_texts: list[tuple[str, str, list[str], int | None, str]],
    ) -> ImplementationSpec:
        # cap evidence to ~12k chars (the LLM budget for this task)
        budget = 12000
        parts: list[str] = []
        index: list[tuple[str, str, int | None]] = []  # chunk_id, section, page
        used = 0
        for i, (chunk_id, text, section_path, page, _ctype) in enumerate(evidence_texts):
            snippet = text[:900]
            if used + len(snippet) > budget:
                break
            parts.append(f"[{i + 1}] ({' › '.join(section_path) or 'front'}) p.{page}\n{snippet}")
            index.append((chunk_id, " › ".join(section_path), page))
            used += len(snippet)

        resp = self.llm.chat(
            [
                LLMMessage(
                    role="system",
                    content=SPEC_PROMPT.format(evidence="\n\n".join(parts)),
                ),
                LLMMessage(role="user", content="Produce the JSON specification."),
            ],
            temperature=0.1,
            max_tokens=self.settings.llm_max_tokens,
        )
        data = self._parse_json(resp.text)
        return self._build_spec(paper_id, paper_title, data, index)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # take the outermost braces if there is prose around
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in LLM response")
        return json.loads(text[start : end + 1])

    @staticmethod
    def _build_spec(
        paper_id: str,
        paper_title: str | None,
        data: dict,
        index: list[tuple[str, str, int | None]],
    ) -> ImplementationSpec:
        requirements: list[SpecRequirement] = []
        for i, r in enumerate(data.get("requirements", [])):
            try:
                rtype = RequirementType(str(r.get("type", "architecture")).lower())
            except ValueError:
                rtype = RequirementType.ARCHITECTURE
            try:
                source = SourceKind(str(r.get("source", "inferred")).lower())
            except ValueError:
                source = SourceKind.INFERRED
            conf = float(r.get("confidence", 0.5))
            ev = r.get("evidence") or []
            chunk_id = section = page = None
            for e in ev:
                try:
                    idx = int(e) - 1
                    if 0 <= idx < len(index):
                        chunk_id, section, page = index[idx]
                        break
                except (TypeError, ValueError):
                    continue
            requirements.append(
                SpecRequirement(
                    id=f"REQ-{i + 1:03d}",
                    requirement=str(r.get("requirement", "")).strip(),
                    type=rtype,
                    source=source,
                    confidence=max(0.0, min(1.0, conf)),
                    implementation_implication=str(
                        r.get("implementation_implication", "")
                    ).strip(),
                    status=RequirementStatus.OPEN,
                    source_section=r.get("source_section") or section,
                    source_page=r.get("source_page") or page,
                    source_chunk_id=chunk_id,
                    source_quote=r.get("source_quote"),
                    source_ref=r.get("source_ref"),
                    notes=str(r.get("notes", "")),
                )
            )
        return ImplementationSpec(
            id=new_id("spec"),
            paper_id=paper_id,
            title=f"Implementation specification — {paper_title or paper_id}",
            summary=str(data.get("summary", "")).strip(),
            requirements=requirements,
            task_decomposition=[str(t) for t in data.get("task_decomposition", [])],
            dependencies=[str(d) for d in data.get("dependencies", [])],
            open_questions=[str(q) for q in data.get("open_questions", [])],
            generated_by="llm",
            created_at=now_iso(),
        )
