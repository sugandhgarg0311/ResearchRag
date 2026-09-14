"""Evidence selection for implementation-spec generation.

The spec generator is only as good as the paper text it reads. This
selector picks the chunks most likely to contain implementation
decisions: tables first (parameters live there), then method/training/
experiment sections, then the rest — in document order — up to ``limit``.

Shared by the API and the CLI so both select identical evidence.
"""

from __future__ import annotations

#: section keywords, in priority order, that indicate implementation content
_METHOD_KEYS = ("method", "model", "training", "experiment", "approach", "retriever", "generator")
_RESULT_KEYS = ("result", "evaluation")


def spec_evidence(ctx, paper_id: str, limit: int = 60) -> list[tuple[str, str, list[str], int | None, str]]:
    """Return up to ``limit`` (chunk_id, text, section_path, page, chunk_type)."""
    chunks = ctx.db.get_chunks(paper_id, kind="child")

    def priority(c: dict) -> int:
        section = " ".join(c.get("section_path", [])).lower()
        ctype = c.get("chunk_type", "")
        if ctype == "table":
            return 0
        if any(k in section for k in _METHOD_KEYS):
            return 1
        if any(k in section for k in _RESULT_KEYS):
            return 2
        return 3

    ordered = sorted(chunks, key=lambda c: (priority(c), c.get("page_start", 0)))
    out = []
    for c in ordered:
        out.append(
            (
                c["id"],
                c["text"],
                c.get("section_path", []),
                c.get("page_start"),
                c.get("chunk_type", "text"),
            )
        )
        if len(out) >= limit:
            break
    return out
