"""Job status endpoints (background work: ingest, spec, implement, eval)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from researchrag.api.deps import get_ctx

router = APIRouter(tags=["jobs"])


@router.get("/jobs")
def list_all_jobs(ctx=Depends(get_ctx)):
    jobs = ctx.db.list_jobs(None)
    for j in jobs:
        raw = ctx.db.kv_get(f"job_result:{j['id']}")
        j["result"] = json.loads(raw) if raw else None
    return {"jobs": jobs}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, ctx=Depends(get_ctx)):
    job = ctx.db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    raw = ctx.db.kv_get(f"job_result:{job_id}")
    job["result"] = json.loads(raw) if raw else None
    return job


@router.get("/papers/{paper_id}/jobs")
def list_jobs(paper_id: str, ctx=Depends(get_ctx)):
    from researchrag.api.deps import require_paper

    require_paper(ctx, paper_id)
    jobs = ctx.db.list_jobs(paper_id)
    for j in jobs:
        raw = ctx.db.kv_get(f"job_result:{j['id']}")
        j["result"] = json.loads(raw) if raw else None
    return {"jobs": jobs}
