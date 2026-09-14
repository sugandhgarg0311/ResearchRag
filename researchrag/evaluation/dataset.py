"""Evaluation dataset: schema + loading.

A dataset is a JSON file:

{
  "paper": "rag_paper.pdf",
  "questions": [
    {
      "id": "q01",
      "question": "...",
      "class": "hyperparameter_lookup",   // question class (12 supported)
      "gold_chunks": ["<chunk id or text-fragment>", ...],
      "gold_answer": "..."                // for answer-level metrics
    }
  ]
}

Gold chunks can be exact chunk ids OR text fragments: fragments are
resolved against the paper's chunks at load time (first chunk containing
the fragment), which makes datasets portable across re-chunking while
ids keep them exact when available.

Question classes (the 12 required by the product spec):
  fact_lookup, definition, methodology, implementation_detail,
  hyperparameter_lookup, comparison, numerical_result, equation_reasoning,
  table_reasoning, figure_reasoning, cross_section_reasoning,
  global_paper
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from researchrag.models.chunk import Chunk

QUESTION_CLASSES = [
    "fact_lookup",
    "definition",
    "methodology",
    "implementation_detail",
    "hyperparameter_lookup",
    "comparison",
    "numerical_result",
    "equation_reasoning",
    "table_reasoning",
    "figure_reasoning",
    "cross_section_reasoning",
    "global_paper",
]


@dataclass
class EvalQuestion:
    id: str
    question: str
    question_class: str
    gold_chunks: list[str] = field(default_factory=list)
    gold_answer: str = ""


@dataclass
class EvalDataset:
    name: str
    paper: str
    questions: list[EvalQuestion] = field(default_factory=list)

    def by_class(self, qclass: str) -> list[EvalQuestion]:
        return [q for q in self.questions if q.question_class == qclass]


def load_dataset(path: Path | str) -> EvalDataset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = [
        EvalQuestion(
            id=str(q.get("id", f"q{i + 1}")),
            question=q["question"],
            question_class=q.get("class", "fact_lookup"),
            gold_chunks=list(q.get("gold_chunks", [])),
            gold_answer=q.get("gold_answer", ""),
        )
        for i, q in enumerate(data.get("questions", []))
    ]
    return EvalDataset(
        name=data.get("name", Path(path).stem),
        paper=data.get("paper", ""),
        questions=questions,
    )


def resolve_gold_chunks(ds: EvalDataset, chunks: list[Chunk]) -> dict[str, set[str]]:
    """Map question id -> set of gold chunk ids, resolving text fragments."""
    by_id = {c.id: c for c in chunks}
    out: dict[str, set[str]] = {}
    for q in ds.questions:
        gold: set[str] = set()
        for ref in q.gold_chunks:
            if ref in by_id:
                gold.add(ref)
                continue
            needle = ref.strip().lower()
            if len(needle) < 12:
                continue
            for c in chunks:
                if needle in c.text.lower():
                    gold.add(c.id)
                    break
        out[q.id] = gold
    return out
