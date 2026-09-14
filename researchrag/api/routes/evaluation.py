"""Evaluation & experiment endpoints (thin wrappers over evaluation.service)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from researchrag.api.deps import get_ctx, require_paper, run_job
from researchrag.evaluation import service

router = APIRouter(tags=["evaluation"])


@router.get("/papers/{paper_id}/eval-datasets")
def list_datasets(paper_id: str, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)
    return {"datasets": service.list_datasets(ctx.settings)}


@router.post("/papers/{paper_id}/evaluate", status_code=202)
def evaluate_paper(
    paper_id: str,
    dataset: str | None = None,
    k: int = 10,
    ctx=Depends(get_ctx),
):
    """Run retrieval + answer metrics over an eval dataset (background job)."""
    require_paper(ctx, paper_id)
    names = service.list_datasets(ctx.settings)
    name = dataset or (names[0] if names else None)
    if not name or name not in names:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No eval dataset '{name or ''}'. Place a JSON dataset in the "
                "evals directory (format: researchrag/evaluation/dataset.py)."
            ),
        )
    job_id = ctx.db.create_job(paper_id, "evaluate")
    run_job(ctx, job_id, service.run_evaluation, ctx, paper_id, name, k)
    return {"job_id": job_id, "status": "queued", "dataset": name}


@router.get("/papers/{paper_id}/evaluations")
def list_evaluations(paper_id: str, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)
    return {"evaluations": ctx.db.list_eval_runs(paper_id)}


@router.get("/papers/{paper_id}/experiments")
def list_experiments(paper_id: str, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)
    return {"experiments": ctx.db.list_experiments(paper_id)}


@router.post("/papers/{paper_id}/experiments", status_code=202)
def run_experiments(paper_id: str, ctx=Depends(get_ctx)):
    """Compare retrieval configurations (chunking × dense/sparse/hybrid × RRF × K)."""
    require_paper(ctx, paper_id)
    names = service.list_datasets(ctx.settings)
    if not names:
        raise HTTPException(
            status_code=404,
            detail="No eval dataset found in the evals directory.",
        )
    job_id = ctx.db.create_job(paper_id, "experiment")
    run_job(ctx, job_id, service.run_experiments, ctx, paper_id, names[0])
    return {"job_id": job_id, "status": "queued", "dataset": names[0]}
