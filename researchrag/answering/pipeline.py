"""Grounded question answering.

Flow:  query → hybrid retrieval → evidence assembly → LLM (or extractive
fallback) → answer + citations + confidence + sufficiency.

The answer always carries:
* the evidence chunks it was built from,
* one citation per supporting passage (paper, section, page range),
* a confidence class (explicit / inferred / external / unknown),
* a sufficiency class (sufficient / partial / insufficient),
* per-stage timings for the observability requirements.
"""

from __future__ import annotations

import logging
import time

from researchrag.answering.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    strip_markers,
)
from researchrag.config import Settings
from researchrag.evidence import EvidenceAssembler, EvidencePackage
from researchrag.llm.base import BaseLLM, LLMError, LLMMessage
from researchrag.llm.extractive import ExtractiveAnswerer
from researchrag.models.retrieval import Answer, RetrievalResult
from researchrag.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)


class AnsweringPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        llm: BaseLLM | None,
        assembler: EvidenceAssembler | None = None,
    ):
        self.settings = settings
        self.retriever = retriever
        self.llm = llm
        self.assembler = assembler or EvidenceAssembler(
            max_tokens=settings.evidence_max_tokens,
            use_parent_context=settings.use_parent_context,
        )
        self.extractive = ExtractiveAnswerer()

    # ------------------------------------------------------------------
    def retrieve_evidence(
        self, question: str, paper_id: str, config=None
    ) -> tuple[RetrievalResult, EvidencePackage]:
        result = self.retriever.retrieve(question, paper_id=paper_id, config=config)
        paper_title = (
            result.evidence[0].chunk.paper_title if result.evidence else None
        )
        package = self.assembler.assemble(result.evidence, paper_title=paper_title)
        return result, package

    def answer(
        self,
        question: str,
        paper_id: str,
        config=None,
    ) -> Answer:
        t0 = time.perf_counter()
        retrieval, package = self.retrieve_evidence(question, paper_id, config)

        if self.llm is None:
            ans = self.extractive.answer(question, package)
        else:
            ans = self._llm_answer(question, package, retrieval)

        if retrieval.stages:
            ans.stages = retrieval.stages
            ans.stages.assembly_ms = package.assembly_ms
            ans.stages.total_ms = (time.perf_counter() - t0) * 1000
        return ans

    # ------------------------------------------------------------------
    def _llm_answer(
        self, question: str, package: EvidencePackage, retrieval
    ) -> Answer:

        evidence_text = "\n\n".join(item.context_text for item in package.items)
        system = SYSTEM_PROMPT.format(evidence=evidence_text)
        t0 = time.perf_counter()
        try:
            resp = self.llm.chat(
                [
                    LLMMessage(role="system", content=system),
                    LLMMessage(role="user", content=USER_PROMPT.format(question=question)),
                ],
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            )
        except LLMError as e:
            log.warning("LLM answer failed (%s); using extractive fallback", e)
            ans = self.extractive.answer(question, package)
            ans.answer = (
                ans.answer
                + f" (LLM unavailable: {type(e).__name__}; extractive fallback used)"
            )
            return ans
        llm_ms = (time.perf_counter() - t0) * 1000
        log.info("LLM answer in %.0fms", llm_ms)

        clean, confidence, sufficiency = strip_markers(resp.text)
        return Answer(
            question=question,
            answer=clean,
            mode="generative",
            model=resp.model,
            citations=[item.citation for item in package.items],
            evidence=[item.chunk for item in package.items],
            sufficiency=sufficiency or "partial",
            confidence=confidence or "strongly_inferred",
            tokens_used=resp.tokens_used,
        )
