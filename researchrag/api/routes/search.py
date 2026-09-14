"""Search endpoints: grounded chat + raw retrieval debugging."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from researchrag.api.deps import get_ctx, require_paper
from researchrag.models.retrieval import Answer, RetrievalConfig, RetrievalResult

router = APIRouter(tags=["search"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    config: RetrievalConfig | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    config: RetrievalConfig | None = None


@router.post("/papers/{paper_id}/chat", response_model=Answer)
def chat(paper_id: str, body: ChatRequest, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)  # 404 guard
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    answerer = ctx.get_answerer(paper_id)
    try:
        return answerer.answer(body.question.strip(), paper_id, config=body.config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answering failed: {e}")


@router.post("/papers/{paper_id}/retrieve", response_model=RetrievalResult)
def retrieve(paper_id: str, body: RetrieveRequest, ctx=Depends(get_ctx)):
    require_paper(ctx, paper_id)  # 404 guard
    retriever = ctx.get_retriever(paper_id)
    try:
        return retriever.retrieve(body.query.strip(), paper_id, config=body.config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
