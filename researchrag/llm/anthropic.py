"""Anthropic Messages API provider (httpx; no SDK needed)."""

from __future__ import annotations

import logging
import time

import httpx

from researchrag.llm.base import BaseLLM, LLMError, LLMMessage, LLMResponse

log = logging.getLogger(__name__)

BASE_URL = "https://api.anthropic.com/v1"
VERSION = "2023-06-01"


class AnthropicLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str,
        timeout_s: float = 180.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.name = f"anthropic/{model}"

    def is_online(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        url = f"{BASE_URL}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": VERSION,
        }
        system = "\n".join(m.content for m in messages if m.role == "system")
        chat_msgs = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_msgs,
        }
        if system:
            body["system"] = system

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(url, json=body, headers=headers)
                dt = (time.perf_counter() - t0) * 1000
                if resp.status_code >= 500 or resp.status_code == 429:
                    last_error = LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    time.sleep(min(2 ** attempt, 8))
                    continue
                if resp.status_code >= 400:
                    raise LLMError(
                        f"Anthropic request failed (HTTP {resp.status_code}): "
                        f"{resp.text[:500]}"
                    )
                data = resp.json()
                text = "".join(
                    b.get("text", "") for b in data.get("content", [])
                    if b.get("type") == "text"
                )
                usage = data.get("usage") or {}
                log.info("Anthropic: %dms, model=%s", dt, self.model)
                return LLMResponse(
                    text=text,
                    model=data.get("model", self.model),
                    tokens_used=usage.get("input_tokens") + usage.get("output_tokens"),
                    finish_reason=data.get("stop_reason"),
                    raw=data,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = LLMError(f"network error: {e}")
                time.sleep(min(2 ** attempt, 8))
        raise LLMError(
            f"Anthropic request failed after {self.max_retries + 1} attempts: {last_error}"
        )
