"""Lightweight, dependency-free sentence segmentation.

Why not spaCy/NLTK? Those pull large models or data files for what we need.
For research papers, a careful rule-based splitter (period + case logic,
abbreviation and decimal handling) is deterministic, fast, and testable.

The function works on *whitespace-normalized* text and returns spans into
that normalized string, so callers can join sentences exactly.
"""

from __future__ import annotations

import re

#: Common abbreviations that must not trigger a sentence break.
_ABBREVIATIONS = {
    "et", "al", "e.g", "i.e", "cf", "vs", "etc", "fig", "figs", "eq", "eqs",
    "sec", "secs", "ch", "chs", "vol", "vols", "no", "nos", "approx", "min",
    "max", "argmax", "argmin", "sup", "sub", "def", "resp", "dept", "prof",
    "dr", "mr", "mrs", "ms", "st", "jr", "sr", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "oct", "nov", "dec", "inc", "ltd", "co",
    "corp", "univ", "university", "institute",
}

_SENT_END = re.compile(r"([.!?]+)(\s+|$)")


def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces, trim."""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split *normalized* text into sentences.

    Returns a list of (sentence_text, start, end) with character offsets
    into the input string.
    """
    text = text.strip()
    if not text:
        return []

    sentences: list[tuple[str, int, int]] = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        m = _SENT_END.search(text, i)
        if not m:
            tail = text[start:].strip()
            if tail:
                s, e = _bounds(text, start)
                sentences.append((text[s:e], s, e))
            break

        end_punct = m.end()  # position after punctuation (before whitespace)
        # Candidate next sentence starts after trailing whitespace.
        j = end_punct
        while j < n and text[j].isspace():
            j += 1

        if j >= n:
            break

        # What follows the terminator?
        next_char = text[j]
        # 1) "..." style: only split after the final dot of the run.
        # 2) Lowercase or digit next -> probably abbreviation/decimal -> no split.
        # 3) Uppercase or opening quote/paren -> likely a new sentence.
        looks_new = (
            next_char.isupper()
            or next_char in "\"'“‘([{"
            or next_char.isdigit()
        )
        prev_word = _preceding_word(text, m.start(1))
        is_abbreviation = prev_word.lower() in _ABBREVIATIONS or _is_single_letter(
            prev_word
        )

        if looks_new and not is_abbreviation and len(prev_word) > 0:
            # Split: sentence runs from `start` to the end of the punctuation.
            s, e = _bounds(text, start, end_punct)
            if s is not None:
                sentences.append((text[s:e], s, e))
            start = j
            i = j
        else:
            i = end_punct

    # Any remainder without terminator.
    tail = text[start:].strip()
    if tail:
        s, e = _bounds(text, start)
        if s is not None and (not sentences or text[s:e] != sentences[-1][0]):
            sentences.append((text[s:e], s, e))

    return sentences


def _is_single_letter(word: str) -> bool:
    return len(word) == 1 and word.isalpha()


def _preceding_word(text: str, end: int) -> str:
    """Word immediately before position `end` (before the punctuation)."""
    i = end - 1
    while i >= 0 and text[i].isalnum():
        i -= 1
    return text[i + 1 : end]


def _bounds(
    text: str, start: int, hard_end: int | None = None
) -> tuple[int | None, int | None]:
    """Trim whitespace/stray punctuation at the edges of a sentence span."""
    if hard_end is None:
        hard_end = len(text)
    s = start
    e = hard_end
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if s >= e:
        return None, None
    return s, e


def sentence_count(text: str) -> int:
    return len(split_sentences(text))
