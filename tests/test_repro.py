"""Unit tests for reproduction metrics (METRICS parsing, honest comparison)."""

from __future__ import annotations

from researchrag.repro.metrics import parse_metrics_from_output, record_run_results


class _FakeDB:
    def __init__(self):
        self.records = []

    def insert_reproduction_record(self, rec):
        self.records.append(rec)
        return f"rec{len(self.records)}"


def test_parse_metrics_line():
    out = "training finished\nMETRICS: {\"em\": 44.5, \"f1\": 50.0, \"note\": \"x\"}\ndone"
    m = parse_metrics_from_output(out)
    assert m["em"] == 44.5 and m["f1"] == 50.0
    assert m["note"] == "x"  # string metrics are kept (qualitative values)


def test_parse_metrics_absent():
    assert parse_metrics_from_output("no metrics here") == {}


def test_parse_metrics_invalid_json():
    assert parse_metrics_from_output("METRICS: {not json}") == {}


def test_record_run_results_computes_difference():
    db = _FakeDB()
    run = {"id": "r1", "metrics": {"em": 44.0, "f1": 49.5}}
    recs = record_run_results(db, "p", run, paper_values={"em": 44.5, "f1": 50.0})
    assert len(recs) == 2
    by_metric = {r["metric"]: r for r in db.records}
    assert by_metric["em"]["difference"] == -0.5
    assert by_metric["f1"]["difference"] == -0.5


def test_missing_paper_value_is_reported_not_fabricated():
    db = _FakeDB()
    run = {"id": "r1", "metrics": {"em": 44.0}}
    record_run_results(db, "p", run, paper_values={})  # paper reports nothing
    rec = db.records[0]
    assert rec["paper_value"] is None
    assert rec["difference"] is None
    assert "not reported" in rec["notes"].lower()


def test_non_numeric_metrics_are_skipped():
    db = _FakeDB()
    run = {"id": "r1", "metrics": {"note": "hello", "em": 1.0}}
    record_run_results(db, "p", run, paper_values={"em": 1.0})
    assert [r["metric"] for r in db.records] == ["em"]
