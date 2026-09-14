#!/usr/bin/env python3
"""Deterministic mock OpenAI-compatible LLM server for Research RAG.

Purpose: exercise the *full generative path* (prompts → LLM → marker
parsing → confidence classification) without any real model or network.
Responses are deterministic functions of the request, so tests and
demos are reproducible.

Endpoints (OpenAI-compatible):
    POST /v1/chat/completions
    GET  /v1/models
    GET  /healthz

Usage:
    python scripts/mock_llm_server.py --port 11435
    RESEARCHRAG_LLM_PROVIDER=openai_compatible \
    RESEARCHRAG_LLM_BASE_URL=http://localhost:11435/v1 \
    RESEARCHRAG_LLM_MODEL=mock researchrag ask paper_... "question"

What it does per request kind (detected from the system prompt):
  * grounded-QA prompt   → verbatim composition from the best-overlapping
                           numbered evidence passages, with [n] citations
                           and CONFIDENCE:/SUFFICIENCY: markers;
  * spec-generation prompt → a small JSON ImplementationSpec with explicit
                           requirements extracted from the evidence;
  * code-file prompt       → echoes the scaffold code (deterministic safe
                           baseline, so the LLM codegen path is exercised
                           without inventing code).
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# request-kind detection
# ---------------------------------------------------------------------------

def _system_of(messages: list[dict]) -> str:
    return next((m["content"] for m in messages if m.get("role") == "system"), "")


def _user_of(messages: list[dict]) -> str:
    return next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")


def _is_qa_prompt(system: str) -> bool:
    return "Evidence passages (numbered):" in system and "CONFIDENCE" in system


def _is_spec_prompt(system: str) -> str:
    return "implementation" in system.lower() and "json" in system.lower()


def _is_code_prompt(system: str) -> str:
    return "Write ONLY the complete contents of the file" in system


# ---------------------------------------------------------------------------
# grounded QA: verbatim composition from the evidence passages
# ---------------------------------------------------------------------------

_PASSAGE_RE = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", re.DOTALL)
_WORDS = re.compile(r"[a-z0-9][a-z0-9._\-]*")


def _overlap(a: str, b: str) -> float:
    wa = {w for w in _WORDS.findall(a.lower()) if len(w) > 2}
    wb = {w for w in _WORDS.findall(b.lower()) if len(w) > 2}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa)


def answer_qa(system: str, user: str) -> str:
    question = re.sub(r"^Question:\s*", "", user.strip())
    evidence = system.split("Evidence passages (numbered):", 1)[1]
    passages = {
        int(num): text.strip()
        for num, text in _PASSAGE_RE.findall(evidence)
    }
    if not passages:
        return (
            "The retrieved evidence is empty, so the question cannot be answered.\n"
            "CONFIDENCE: unknown\nSUFFICIENCY: insufficient"
        )

    best = sorted(
        passages.items(), key=lambda kv: -_overlap(kv[1], question)
    )
    top_score = _overlap(best[0][1], question)
    if top_score < 0.15:
        return (
            "The paper does not specify this in the retrieved sections; the "
            "question may concern content outside the paper's scope.\n"
            "CONFIDENCE: unknown\nSUFFICIENCY: insufficient"
        )

    parts: list[str] = []
    used: list[int] = []
    for idx, text in best[:2]:
        # take the single best sentence for tightness
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentence = max(sentences, key=lambda s: _overlap(s, question))
        parts.append(sentence.strip().rstrip(".") + f" [{idx}]")
        used.append(idx)
    confidence = "explicit" if top_score >= 0.3 else "strongly_inferred"
    sufficiency = "sufficient" if top_score >= 0.3 else "partial"
    return (
        " ".join(parts)
        + f"\nCONFIDENCE: {confidence}\nSUFFICIENCY: {sufficiency}"
    )


# ---------------------------------------------------------------------------
# spec generation: minimal explicit-requirements JSON
# ---------------------------------------------------------------------------

_SPEC_PATTERNS = [
    (r"learning rate\D{0,24}?(?:of|:|,)?\s*(1[.]?\d*[eE][\-+]\d{1,2}|\d+[\.,]\d*)",
     "hyperparameter", "Learning rate: {v}", "Set the optimizer learning rate to this value."),
    (r"(AdamW|Adam|SGD|RMSprop)\b",
     "optimizer", "Optimizer: {v}", "Instantiate the named optimizer."),
    (r"batch size\D{0,24}?(?:of|:|,)?\s*(\d+)",
     "hyperparameter", "Batch size: {v}", "Train with this batch size."),
    (r"(\d+)\s+(?:encoder|decoder)\s+layers",
     "architecture", "{v} encoder/decoder layers", "Build the model with this layer count."),
    (r"(A100|V100|P100|H100|T4)\b",
     "hardware", "Hardware: {v} GPU", "Target this hardware."),
    (r"top[- ]?(\d{1,6})\s+documents",
     "inference", "Top-{v} documents at retrieval", "Retrieve this many documents per query."),
    (r"(\d{1,9}[M]?)\s+documents",
     "dataset", "Corpus of {v} documents", "Index this corpus."),
]


def answer_spec(system: str, user: str) -> str:
    evidence = system.split("Evidence passages:", 1)[1] if "Evidence passages:" in system else system
    requirements: list[dict] = []
    seen: set[str] = set()
    for i, line in enumerate(evidence.splitlines()):
        for pattern, rtype, template, implication in _SPEC_PATTERNS:
            m = re.search(pattern, line, re.IGNORECASE)
            if not m:
                continue
            req_text = template.format(v=m.group(1).strip())
            if req_text.lower() in seen:
                continue
            seen.add(req_text.lower())
            requirements.append(
                {
                    "requirement": req_text,
                    "type": rtype,
                    "source": "explicit",
                    "confidence": 0.9,
                    "implementation_implication": implication,
                    "source_section": None,
                    "source_page": None,
                    "evidence": [i + 1],
                    "source_quote": line.strip()[:160],
                    "source_ref": None,
                    "notes": "mock-extracted",
                }
            )
            if len(requirements) >= 20:
                break
        if len(requirements) >= 20:
            break
    spec = {
        "summary": "Mock-generated implementation specification (deterministic offline stand-in).",
        "requirements": requirements,
        "task_decomposition": [
            "Set up environment",
            "Implement data pipeline",
            "Implement model",
            "Implement training loop",
            "Implement evaluation",
        ],
        "dependencies": [],
        "open_questions": [
            "Mock mode: only pattern-extracted requirements are listed; review the paper's "
            "methods and appendices for anything absent.",
        ],
    }
    return json.dumps(spec, indent=1)


# ---------------------------------------------------------------------------
# code file: echo the scaffold (safe deterministic baseline)
# ---------------------------------------------------------------------------

def answer_code(system: str, user: str) -> str:
    marker = "Scaffold version of"
    if marker in system:
        scaffold = system.split(marker, 1)[1].strip()
        # drop the trailing prompt instruction line if present
        lines = [ln for ln in scaffold.splitlines() if not ln.strip().startswith("Return the file")]
        return "\n".join(lines).strip() + "\n"
    return "```python\n# mock: no scaffold available\n```\n"


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # keep stdout clean
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": "mock"}], "object": "list"})
        elif self.path == "/healthz":
            self._send(200, {"status": "ok", "name": "researchrag-mock-llm"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return
        messages = body.get("messages", [])
        system = _system_of(messages)
        user = _user_of(messages)

        if _is_qa_prompt(system):
            text = answer_qa(system, user)
        elif _is_spec_prompt(system):
            text = answer_spec(system, user)
        elif _is_code_prompt(system):
            text = answer_code(system, user)
        else:
            text = "Mock LLM: no handler for this prompt type."

        content_tokens = max(1, len(text) // 4)
        prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        self._send(
            200,
            {
                "id": "mock-0",
                "object": "chat.completion",
                "model": body.get("model", "mock"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": content_tokens,
                    "total_tokens": prompt_tokens + content_tokens,
                },
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11435)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock LLM server on http://{args.host}:{args.port}/v1 (model: mock)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
