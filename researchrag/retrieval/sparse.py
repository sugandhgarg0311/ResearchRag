"""Sparse retrieval: pure-Python Okapi BM25.

Design decisions
----------------
* **Full-corpus index per paper.** Every child chunk of the paper is indexed
  (all pages, all block types). Page numbers are provenance metadata on the
  chunks — they are NEVER a retrieval boundary. Nothing about this index
  limits coverage to one page or a window of pages.
* **Okapi BM25** (Robertson & Walker):  score(q,d) = Σ_idf(qi) ·
  f(qi,d)(k1+1) / (f(qi,d) + k1(1 - b + b·|d|/avgdl)).
  Defaults k1=1.5, b=0.75 (the standard, published values).
* **Tokenizer keeps identifiers intact**: ``RAG-Token`` → ``rag-token``,
  ``state-of-the-art`` → ``state-of-the-art``, ``top-k`` → ``top-k`` —
  the exact technical terms, variable names and hyperparameter notation
  that sparse retrieval exists to catch.
* Zero native dependencies; trivially unit-testable; persisted as JSON so
  ingestion is not re-paid at query time.

Why not rank_bm25 / Elasticsearch / Qdrant BM25?
  rank_bm25 is the same math with less control; ES is a whole service;
  Qdrant's engine-side BM25 couples the index to a specific server version.
  A 60-line explicit implementation is auditable and testable (the project
  standard).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._\-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SparseHit:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.df: dict[str, int] = {}
        self._inv: dict[str, dict[int, int]] = {}  # term -> {doc_idx: freq}

    # ------------------------------------------------------------------
    def add_document(self, chunk_id: str, text: str) -> None:
        tokens = tokenize(text)
        idx = len(self.doc_ids)
        self.doc_ids.append(chunk_id)
        self.doc_tokens.append(tokens)
        self.doc_len.append(len(tokens))
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        for t in freq:
            self.df[t] = self.df.get(t, 0) + 1
            self._inv.setdefault(t, {})[idx] = freq[t]
        n = len(self.doc_len)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0

    def build(self, documents: Sequence[tuple[str, str]]) -> BM25Index:
        for chunk_id, text in documents:
            self.add_document(chunk_id, text)
        return self

    # ------------------------------------------------------------------
    def search(self, query: str, top_n: int = 50) -> list[SparseHit]:
        if not self.doc_ids:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        n = len(self.doc_ids)
        scores: dict[int, float] = {}
        for t in set(q_tokens):
            posting = self._inv.get(t)
            if not posting:
                continue
            idf = math.log(1.0 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for idx, f in posting.items():
                dl = self.doc_len[idx] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[idx] = scores.get(idx, 0.0) + idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        return [
            SparseHit(chunk_id=self.doc_ids[idx], score=score)
            for idx, score in ranked
            if score > 0
        ]

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "docs": len(self.doc_ids),
            "vocab": len(self.df),
            "avgdl": round(self.avgdl, 2),
        }

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "df": self.df,
            "inv": self._inv,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BM25Index:
        idx = cls(k1=d.get("k1", 1.5), b=d.get("b", 0.75))
        idx.doc_ids = d["doc_ids"]
        idx.doc_len = d["doc_len"]
        idx.avgdl = d["avgdl"]
        idx.df = d["df"]
        # JSON round-trip turns dict keys into strings; restore int doc indexes
        idx._inv = {
            t: {int(k): v for k, v in posting.items()} for t, posting in d["inv"].items()
        }
        idx.doc_tokens = []
        return idx


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def index_path_for(data_dir: Path, paper_id: str) -> Path:
    return Path(data_dir) / "index" / f"bm25_{paper_id}.json"


def save_index(idx: BM25Index, data_dir: Path, paper_id: str) -> Path:
    path = index_path_for(data_dir, paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(idx.to_dict()), encoding="utf-8")
    return path


def load_index(data_dir: Path, paper_id: str) -> BM25Index | None:
    path = index_path_for(data_dir, paper_id)
    if not path.exists():
        return None
    try:
        return BM25Index.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def delete_index(data_dir: Path, paper_id: str) -> None:
    path = index_path_for(data_dir, paper_id)
    if path.exists():
        path.unlink()
