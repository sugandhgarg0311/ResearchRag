from __future__ import annotations

import logging

from researchrag.config import Settings
from researchrag.llm.anthropic import AnthropicLLM
from researchrag.llm.base import BaseLLM
from researchrag.llm.openai_compatible import OpenAICompatibleLLM

log = logging.getLogger(__name__)


def make_llm(settings: Settings) -> BaseLLM | None:
    """Build the configured LLM provider, or None when 'offline' / unusable.

    A None return is a *normal* state (offline mode), not an error: the
    answering pipeline then uses the extractive fallback and labels its
    answers accordingly.
    """
    provider = (settings.llm_provider or "offline").lower()
    if provider == "offline":
        log.info("LLM provider: offline (extractive answering only)")
        return None
    if provider == "anthropic":
        if not settings.llm_api_key:
            log.warning("Anthropic provider selected but no API key set — offline mode")
            return None
        return AnthropicLLM(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    if provider == "openai_compatible":
        llm = OpenAICompatibleLLM(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
        if not llm.is_online():
            log.warning(
                "OpenAI-compatible provider selected but no API key and no local "
                "base URL set — offline mode"
            )
            return None
        return llm
    raise ValueError(f"Unknown LLM provider: {provider}")
