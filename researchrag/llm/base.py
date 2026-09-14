"""LLM provider contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMMessage:
    role: str  # system | user | assistant
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_used: int | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """Provider failure after retries (timeout / 5xx / auth)."""


class BaseLLM:
    name: str = "base"

    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        raise NotImplementedError

    def is_online(self) -> bool:
        """True when this provider can actually reach a model."""
        return True
