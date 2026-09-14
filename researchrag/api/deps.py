"""Shared API dependencies: app context, paper resolution, job helpers."""

from __future__ import annotations

import inspect
import logging
import threading

from fastapi import HTTPException, Request

from researchrag.app import AppContext
from researchrag.config import masked_dict

log = logging.getLogger(__name__)


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def require_paper(ctx: AppContext, paper_id: str) -> dict:
    paper = ctx.db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")
    if paper.get("status") != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Paper is not ready (status={paper.get('status')}, error={paper.get('error')})",
        )
    return paper


def run_job(ctx: AppContext, job_id: str, fn, *args, **kwargs) -> None:
    """Run a long operation in a worker thread, tracking it as a job."""

    def worker() -> None:
        ctx.db.update_job(job_id, status="running", progress=0.0)
        try:
            accepts_progress = "progress" in inspect.signature(fn).parameters
            if accepts_progress:
                kwargs["progress"] = lambda p, d: ctx.db.update_job(job_id, progress=p, detail=d)
            result = fn(*args, **kwargs)
            ctx.db.update_job(job_id, status="done", progress=1.0, detail="complete")
            if isinstance(result, dict):
                import json

                ctx.db.kv_set(f"job_result:{job_id}", json.dumps(result, default=str))
        except Exception as e:
            log.exception("job %s failed", job_id)
            ctx.db.update_job(job_id, status="failed", error=f"{type(e).__name__}: {e}")

    t = threading.Thread(target=worker, daemon=True, name=f"job-{job_id}")
    t.start()


def config_status(ctx: AppContext) -> dict:
    import researchrag

    return {
        "version": researchrag.__version__,
        "parser": ctx.parser.name,
        "embedding_model": ctx.embedder.model_name,
        "embedding_fallback": ctx._embedder_fell_back,
        "reranker": ctx.reranker.name,
        "llm": ctx.llm.name if ctx.llm else "offline",
        "chunk_strategy": ctx.chunker.name,
        "settings": masked_dict(ctx.settings),
    }
