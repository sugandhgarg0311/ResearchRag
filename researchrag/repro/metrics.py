"""Reproduction records: paper values vs. our values.

The system NEVER fabricates reproduction results. A record is created only
when both a paper value (extracted from the paper's reported results) and
an "our value" (parsed from an actual execution's METRICS output) exist —
or when either is missing, in which case the record explicitly says so.
Missing information is a *feature*: it tells the user exactly what the
paper omitted or what the run did not produce.
"""

from __future__ import annotations

import json
import re
from typing import Any

from researchrag.storage.database import Database

METRICS_LINE = re.compile(r"^\s*METRICS\s*:\s*(\{.*\})\s*$", re.MULTILINE)


def parse_metrics_from_output(stdout: str) -> dict[str, Any]:
    """Extract a METRICS: {json} line from execution output."""
    m = METRICS_LINE.search(stdout or "")
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
        return {k: v for k, v in data.items() if isinstance(v, (int, float, str))}
    except json.JSONDecodeError:
        return {}


def record_run_results(
    db: Database,
    paper_id: str,
    run: dict,
    paper_values: dict[str, float] | None = None,
    notes: str = "",
) -> list[dict]:
    """Create reproduction records for one run's metrics.

    paper_values: metric -> value as reported in the paper (empty/None when
    the paper doesn't report a comparable number — records then show
    difference=null, not a fabricated gap).
    """
    our = run.get("metrics", {}) or {}
    paper_values = paper_values or {}
    records = []
    for metric, our_value in our.items():
        if not isinstance(our_value, (int, float)):
            continue
        paper_value = paper_values.get(metric)
        difference = (
            float(our_value) - float(paper_value) if paper_value is not None else None
        )
        rec_id = db.insert_reproduction_record(
            {
                "paper_id": paper_id,
                "metric": str(metric),
                "paper_value": paper_value,
                "our_value": float(our_value),
                "difference": difference,
                "dataset": run.get("dataset"),
                "seed": str(run.get("seed", "")) or None,
                "hardware": run.get("hardware"),
                "config": {
                    "run_id": run.get("id"),
                    "run_kind": run.get("kind"),
                    "command": run.get("command"),
                },
                "notes": notes or (
                    "Paper value not reported in the paper — difference not computable."
                    if paper_value is None
                    else ""
                ),
            }
        )
        records.append({"id": rec_id, "metric": metric})
    return records
