"""Offline extractive answering (no LLM required).

A *real* fallback, not a fake: when no LLM provider is configured (no API
key, no local server), questions are still answered by composing the most
relevant sentences from the retrieved evidence, verbatim, with citations.
The answer is labeled ``mode="extractive"`` so the UI and users are always
told they are reading quoted paper text, not a generative synthesis.

This keeps every workflow in the product (QA, specs via the rule-based
extractor, evaluation) fully functional offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from researchrag.evidence import EvidencePackage
from researchrag.models.retrieval import Answer
from researchrag.tokens import STOP_WORDS, get_token_counter


@dataclass
class ExtractiveAnswerResult:
    answer: str
    confidence: str
    sufficiency: str


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _sentence_overlap(sentence: str, query: str) -> float:
    def terms(text: str) -> set[str]:
        return {
            t
            for t in re.findall(r"[a-z0-9][a-z0-9._\-]*", text.lower())
            if len(t) > 2 and t not in STOP_WORDS
        }

    q_terms = terms(query)
    if not q_terms:
        return 0.0
    s_terms = terms(sentence)
    if not s_terms:
        return 0.0
    return len(q_terms & s_terms) / len(q_terms)


class ExtractiveAnswerer:
    def __init__(self, max_sentences: int = 5, max_tokens: int = 700):
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        self._counter = get_token_counter()

    def answer(self, question: str, evidence: EvidencePackage) -> Answer:
        # rank sentences across all evidence items by overlap with the query
        scored: list[tuple[float, int, str, int]] = []  # (score, rank, text, item_idx)
        for item_idx, item in enumerate(evidence.items):
            sentences = _SENT_END.split(item.chunk.text.strip())
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                scored.append(
                    (_sentence_overlap(sent, question), item_idx, sent, 0)
                )
        scored.sort(key=lambda x: -x[0])
        best = scored[: self.max_sentences]

        if not best or best[0][0] <= 0.05:
            return Answer(
                question=question,
                answer=(
                    "The retrieved sections of this paper do not contain information "
                    "that answers this question. The relevant content may be missing "
                    "from the paper, or the question may be outside the paper's scope."
                ),
                mode="extractive",
                model=None,
                citations=[i.citation for i in evidence.items],
                evidence=[i.chunk for i in evidence.items],
                sufficiency="insufficient",
                confidence="unknown",
            )

        used = 0
        parts: list[str] = []
        cited_items: set[int] = set()
        for score, item_idx, sent, _ in best:
            if score <= 0.05:
                break  # sentences are sorted; stop before pulling in unrelated text
            cost = self._counter.count(sent)
            if used + cost > self.max_tokens:
                break
            parts.append(sent)
            cited_items.add(item_idx)
            used += cost

        sufficiency = "sufficient" if best[0][0] >= 0.35 else "partial"
        confidence = "explicit" if best[0][0] >= 0.35 else "strongly_inferred"
        return Answer(
            question=question,
            answer=" ".join(parts),
            mode="extractive",
            model=None,
            citations=[evidence.items[i].citation for i in sorted(cited_items)],
            evidence=[evidence.items[i].chunk for i in sorted(cited_items)],
            sufficiency=sufficiency,
            confidence=confidence,
        )
