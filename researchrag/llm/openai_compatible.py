"""OpenAI-compatible chat provider.

Works with any ``/chat/completions`` server: OpenAI, OpenRouter, Groq,
local Ollama (``http://localhost:11434/v1``), vLLM, LM Studio, …
"""

from __future__ import annotations

import logging
import time

import httpx

from researchrag.llm.base import BaseLLM, LLMError, LLMMessage, LLMResponse

log = logging.getLogger(__name__)

DEFAULT_BASE_URLS = {
    "openai_compatible": "https://api.openai.com/v1",
}


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 180.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URLS["openai_compatible"]).rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.name = f"openai_compatible/{model}"

    def is_online(self) -> bool:
        if not self.api_key and self.base_url.startswith("http://localhost"):
            return True  # local servers don't need keys
        return bool(self.api_key)

    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(url, json=body, headers=headers)
                dt = (time.perf_counter() - t0) * 1000
                if resp.status_code >= 500 or resp.status_code == 429:
                    last_error = LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    log.warning("LLM attempt %d failed: %s", attempt + 1, last_error)
                    time.sleep(min(2 ** attempt, 8))
                    continue
                if resp.status_code >= 400:
                    raise LLMError(
                        f"LLM request failed (HTTP {resp.status_code}): "
                        f"{resp.text[:500]}"
                    )
                data = resp.json()
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
                usage = data.get("usage") or {}
                log.info(
                    "LLM: %dms, %d completion tokens (model=%s)",
                    dt,
                    usage.get("completion_tokens"),
                    self.model,
                )
                return LLMResponse(
                    text=text,
                    model=data.get("model", self.model),
                    tokens_used=usage.get("completion_tokens")
                    + usage.get("prompt_tokens"),
                    finish_reason=choice.get("finish_reason"),
                    raw=data,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = LLMError(f"network error: {e}")
                log.warning("LLM attempt %d network error: %s", attempt + 1, e)
                time.sleep(min(2 ** attempt, 8))
        raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {last_error}")
