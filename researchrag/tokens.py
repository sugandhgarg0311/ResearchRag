"""Token counting with a graceful fallback.

Primary: tiktoken's ``cl100k_base`` (the GPT-4 family tokenizer) — accurate
budgets that match what modern LLMs count.

Fallback: a deterministic heuristic (1 token ≈ 4 characters of English text,
±10% for research prose). This keeps the pipeline usable in environments
where tiktoken's runtime download is blocked; the chunking logic is
identical, only the budgets shift slightly.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Protocol


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class TiktokenCounter:
    name = "tiktoken/cl100k_base"

    def __init__(self) -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


class HeuristicCounter:
    """~4 characters per token (English). encode/decode operate on characters
    so word-level splitting and overlap logic still work uniformly."""

    name = "heuristic/char4"
    _CHARS_PER_TOKEN = 4

    def count(self, text: str) -> int:
        return max(1, math.ceil(len(text) / self._CHARS_PER_TOKEN))

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


#: Minimal English stopword list shared by term-overlap scoring (extractive
#: answering, answer metrics). Function words carry no content signal;
#: counting them as "overlap" makes unrelated sentences look supported.
STOP_WORDS = frozenset(
    """
    a an the and or but if then else of in on at to from by with without
    for as is are was were be been being has have had do does did will
    would can could should may might must not no nor so too very
    this that these those it its they them their he she we you i
    about into over under between each such than when where which who
    also more most less least often always never
    """.split()
)


@lru_cache
def get_token_counter() -> TokenCounter:
    try:
        return TiktokenCounter()
    except Exception:
        return HeuristicCounter()
