"""Research RAG API — FastAPI application factory.

Run:  uvicorn researchrag.api.app:app --host 0.0.0.0 --port 8000
     (or: researchrag serve)

Serves:
* /api/*          — the platform API (papers, chat, specs, code, evals)
* /healthz        — liveness
* /               — the built frontend (if frontend/dist exists)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from researchrag.api import deps  # noqa: F401  (ensures import order)
from researchrag.api.routes import evaluation, implementations, jobs, papers, search, specs
from researchrag.app import get_app_context

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx = get_app_context()
        app.state.ctx = ctx
        log.info("Research RAG API ready")
        yield

    app = FastAPI(
        title="Research RAG",
        version=__import__("researchrag").__version__,
        description=(
            "Research paper understanding, evidence retrieval, and "
            "implementation reproduction."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- health ---------------------------------------------------------
    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/api/health")
    def health(request: Request):
        from researchrag.api.deps import config_status

        return {"status": "ok", **config_status(deps.get_ctx(request))}

    # ---- routers ---------------------------------------------------------
    app.include_router(papers.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(specs.router, prefix="/api")
    app.include_router(implementations.router, prefix="/api")
    app.include_router(evaluation.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")

    # ---- frontend (static) ------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        dist = Path(__import__("researchrag").__file__).parent.parent / "frontend" / "dist"
        index_file = dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(
            {
                "name": "Research RAG",
                "docs": "/docs",
                "note": "Frontend not built. Run: cd frontend && npm install && npm run build",
            }
        )

    dist = Path(__import__("researchrag").__file__).parent.parent / "frontend" / "dist"
    if (dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    # SPA deep links in production mode: any non-API path serves index.html
    # (registered last so /api, /assets, /docs keep their routes).
    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str):
        if path.startswith(("api/", "docs", "openapi.json", "assets/")):
            raise HTTPException(status_code=404)
        index_file = dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend not built")

    return app


app = create_app()
