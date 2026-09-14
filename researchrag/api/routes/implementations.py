"""Code generation, validation, and execution endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from researchrag.api.deps import get_ctx, require_paper, run_job
from researchrag.codegen.generator import CodeGenerator
from researchrag.execution.runner import run_command
from researchrag.repro.metrics import parse_metrics_from_output, record_run_results
from researchrag.spec.evidence import spec_evidence
from researchrag.spec.loader import spec_from_db
from researchrag.storage.database import new_id, now_iso

router = APIRouter(tags=["implementations"])


class GenerateRequest(BaseModel):
    spec_id: str | None = None
    llm: bool = True  # use LLM fill-in when available (else scaffold only)


class RunRequest(BaseModel):
    kind: str = "smoke"  # smoke | tests | train
    timeout_s: int | None = None


def _generator(ctx, use_llm: bool):
    llm = ctx.llm if use_llm else None
    return CodeGenerator(ctx.settings, ctx.db, llm)


def _generate_and_validate(ctx, paper_id: str, spec_id: str | None, use_llm: bool) -> dict:
    if spec_id is None:
        specs = ctx.db.list_specs(paper_id)
        if not specs:
            raise HTTPException(status_code=400, detail="No spec found; generate one first")
        spec_id = specs[0]["id"]
    spec = spec_from_db(ctx, spec_id)
    if spec is None or spec.paper_id != paper_id:
        raise HTTPException(status_code=404, detail="Spec not found for this paper")

    generator = _generator(ctx, use_llm)
    evidence = spec_evidence(ctx, paper_id, limit=14)
    impl_id, validation, project_dir = generator.generate_and_validate(spec, evidence)
    return {
        "implementation_id": impl_id,
        "validation": {
            "ok": validation.ok,
            "stages": [s.__dict__ for s in validation.stages],
            "errors": validation.errors,
        },
        "files": len(ctx.db.get_implementation(impl_id)["files"]),
    }


@router.get("/papers/{paper_id}/implementations")
def list_implementations(paper_id: str, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)
    return {"implementations": ctx.db.list_implementations(paper_id)}


@router.post("/papers/{paper_id}/implementations", status_code=202)
def generate_implementation(
    paper_id: str, body: GenerateRequest | None = None, ctx=Depends(get_ctx)
):
    require_paper(ctx, paper_id)
    spec_id = body.spec_id if body else None
    use_llm = body.llm if body else True
    job_id = ctx.db.create_job(paper_id, "implement")
    run_job(ctx, job_id, _generate_and_validate, ctx, paper_id, spec_id, use_llm)
    return {"job_id": job_id, "status": "queued"}


@router.get("/implementations/{impl_id}")
def get_implementation(impl_id: str, ctx=Depends(get_ctx)):
    impl = ctx.db.get_implementation(impl_id)
    if impl is None:
        raise HTTPException(status_code=404, detail="Implementation not found")
    # don't ship full file contents in the listing; metadata only
    files = impl.pop("files")
    impl["files"] = [{"path": f["path"], "kind": f["kind"], "size": len(f["content"])} for f in files]
    impl["project_dir"] = str(_project_dir(ctx, impl_id))
    return impl


@router.get("/implementations/{impl_id}/files/{file_path:path}")
def get_file(impl_id: str, file_path: str, ctx=Depends(get_ctx)):
    impl = ctx.db.get_implementation(impl_id)
    if impl is None:
        raise HTTPException(status_code=404, detail="Implementation not found")
    for f in impl["files"]:
        if f["path"] == file_path:
            return {"path": f["path"], "content": f["content"], "kind": f["kind"]}
    raise HTTPException(status_code=404, detail="File not found")


def _project_dir(ctx, impl_id: str) -> Path:
    return ctx.settings.artifacts_root / "implementations" / impl_id


@router.post("/implementations/{impl_id}/runs", status_code=202)
def run_implementation(impl_id: str, body: RunRequest | None = None, ctx=Depends(get_ctx)):
    impl = ctx.db.get_implementation(impl_id)
    if impl is None:
        raise HTTPException(status_code=404, detail="Implementation not found")
    kind = (body.kind if body else "smoke") or "smoke"
    if kind not in ("smoke", "tests", "train"):
        raise HTTPException(status_code=400, detail="kind must be smoke|tests|train")
    timeout = (body.timeout_s if body and body.timeout_s else None) or ctx.settings.execution_timeout_s
    job_id = ctx.db.create_job(impl["paper_id"], "run")
    run_job(ctx, job_id, _run, ctx, impl_id, kind, timeout)
    return {"job_id": job_id, "status": "queued"}


def _run(ctx, impl_id: str, kind: str, timeout: int) -> dict:
    impl = ctx.db.get_implementation(impl_id)
    project_dir = _project_dir(ctx, impl_id)
    # ensure files exist on disk (they may have been cleaned)
    for f in impl["files"]:
        target = project_dir / f["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f["content"], encoding="utf-8")
        if f["kind"] == "script":
            target.chmod(0o755)

    if kind == "smoke":
        cmd = ["bash", "run_smoke.sh"]
    elif kind == "tests":
        cmd = ["bash", "-c", "python tests/test_smoke.py"]
    else:  # train
        pkg = next((d for d in (project_dir / "src").iterdir() if d.is_dir()), None)
        module = pkg.name.replace("-", "_") if pkg else "project"
        cmd = ["bash", "-c", f"python -m {module}.train --config config.yaml"]

    log_path = project_dir / "logs" / f"run_{new_id('run')[-6:]}.log"
    result = run_command(
        cmd,
        cwd=project_dir,
        timeout_s=timeout,
        memory_mb=ctx.settings.execution_memory_mb,
        log_path=log_path,
    )
    metrics = parse_metrics_from_output(result.stdout)
    run_record = {
        "id": new_id("run"),
        "paper_id": impl["paper_id"],
        "implementation_id": impl_id,
        "kind": kind,
        "status": "ok" if result.returncode == 0 else ("timeout" if result.timed_out else "failed"),
        "command": cmd,
        "log_path": str(log_path),
        "metrics": metrics,
        "started_at": now_iso(),
        "finished_at": now_iso(),
        "duration_ms": int(result.duration_ms),
        "error": result.stderr[-2000:] if result.returncode != 0 else None,
    }
    ctx.db.insert_run(run_record)
    # reproduction records: compare with paper values when available
    if metrics:
        paper_values = _paper_values(ctx, impl["paper_id"])
        record_run_results(ctx.db, impl["paper_id"], run_record, paper_values)
    return {"run": run_record}


def _paper_values(ctx, paper_id: str) -> dict[str, float]:
    """Paper-reported metric values recorded earlier via the reproduction
    endpoint (or empty — never guessed)."""
    raw = ctx.db.kv_get(f"paper_values:{paper_id}")
    return json.loads(raw) if raw else {}


@router.get("/papers/{paper_id}/runs")
def list_runs(paper_id: str, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)
    return {"runs": ctx.db.list_runs(paper_id)}


# ---------------------------------------------------------------------------
# Reproduction comparison
# ---------------------------------------------------------------------------


class ReproMetric(BaseModel):
    metric: str
    paper_value: float | None = None
    our_value: float | None = None
    dataset: str | None = None
    seed: str | None = None
    notes: str | None = None


@router.get("/papers/{paper_id}/reproduction")
def reproduction(paper_id: str, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)
    records = ctx.db.list_reproduction_records(paper_id)
    return {"records": records}


@router.post("/papers/{paper_id}/reproduction")
def add_reproduction(paper_id: str, metrics: list[ReproMetric], ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)
    created = []
    for m in metrics:
        difference = (
            float(m.our_value) - float(m.paper_value)
            if m.paper_value is not None and m.our_value is not None
            else None
        )
        rec_id = ctx.db.insert_reproduction_record(
            {
                "paper_id": paper_id,
                "metric": m.metric,
                "paper_value": m.paper_value,
                "our_value": m.our_value,
                "difference": difference,
                "dataset": m.dataset,
                "seed": m.seed,
                "hardware": None,
                "config": {},
                "notes": m.notes or "",
            }
        )
        created.append({"id": rec_id, "metric": m.metric})
    return {"created": created}
