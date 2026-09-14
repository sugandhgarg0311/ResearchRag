"""Evaluation & experiment services (shared by the API and the CLI).

These are the single implementations of the two long-running analysis
jobs; the API runs them in background threads, the CLI runs them inline.
No logic is duplicated in the routes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from researchrag.config import Settings
from researchrag.evaluation.dataset import load_dataset
from researchrag.evaluation.experiments import run_retrieval_experiment
from researchrag.evaluation.runner import evaluate_answers, evaluate_retrieval
from researchrag.models.retrieval import RetrievalConfig
from researchrag.retrieval.sparse import BM25Index
from researchrag.storage.database import new_id, now_iso

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def dataset_dir(settings: Settings) -> Path:
    d = Path(settings.evals_root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_datasets(settings: Settings) -> list[str]:
    return sorted(p.name for p in dataset_dir(settings).glob("*.json"))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(ctx, paper_id: str, name: str, k: int = 10) -> dict:
    """Run retrieval + answer metrics over an eval dataset; persist an eval run."""
    ds = load_dataset(dataset_dir(ctx.settings) / name)
    from researchrag.models.chunk import Chunk

    chunk_objs = [Chunk.from_row(c) for c in ctx.db.get_chunks(paper_id, kind="child")]

    retriever = ctx.get_retriever(paper_id)

    def retrieve_fn(query: str, cfg: RetrievalConfig):
        return retriever.retrieve(query, paper_id, config=cfg)

    retrieval = evaluate_retrieval(ds, chunk_objs, retrieve_fn, k=k)
    answerer = ctx.get_answerer(paper_id, use_llm=False)
    answers = evaluate_answers(ds, answerer, paper_id, chunk_objs)

    summary = {
        "dataset": name,
        "k": k,
        "retrieval": retrieval["metrics"],
        "answers": answers["metrics"],
        "by_class": retrieval["by_class"],
        "n_questions": len(ds.questions),
    }
    run_id = new_id("eval")
    ctx.db.insert_eval_run(
        {
            "id": run_id,
            "paper_id": paper_id,
            "dataset": name,
            "config": {"k": k, "chunk_strategy": ctx.chunker.name},
            "metrics": summary,
            "per_question": {
                "retrieval": retrieval["per_question"],
                "answers": answers["per_question"],
            },
            "created_at": now_iso(),
        }
    )
    return {
        "eval_run_id": run_id,
        "summary": summary,
        "per_question": {
            "retrieval": retrieval["per_question"],
            "answers": answers["per_question"],
        },
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_experiments(ctx, paper_id: str, name: str) -> dict:
    """Compare chunking × retrieval × RRF configurations (in-memory)."""
    import numpy as np

    from researchrag.chunking.fixed_size import FixedSizeChunker
    from researchrag.chunking.section_aware import SectionAwareChunker
    from researchrag.models.chunk import Chunk
    from researchrag.models.document import Block, BlockType, Page, Paper

    ds = load_dataset(dataset_dir(ctx.settings) / name)
    rows = [dict(b) for b in ctx.db.get_blocks(paper_id)]

    # Reconstruct the structured Paper from stored blocks.
    blocks: list[Block] = []
    for b in rows:
        try:
            btype = BlockType(b["block_type"])
        except ValueError:
            btype = BlockType.TEXT
        blocks.append(
            Block(
                id=b["id"],
                paper_id=paper_id,
                block_type=btype,
                text=b["text"],
                page=b["page"],
                page_end=b.get("page_end"),
                order=b.get("order", 0),
                section_path=(
                    json.loads(b["section_path_json"]) if b.get("section_path_json") else []
                ),
                section_id=b.get("section_id"),
                bbox=json.loads(b["bbox_json"]) if b.get("bbox_json") else None,
                heading_level=b.get("heading_level"),
                caption=b.get("caption"),
            )
        )
    pages: dict[int, Page] = {}
    for blk in blocks:
        pages.setdefault(blk.page, Page(number=blk.page)).blocks.append(blk)
    paper_row = ctx.db.get_paper(paper_id)
    paper = Paper(
        id=paper_id,
        filename=paper_row.get("filename", ""),
        title=paper_row.get("title"),
        page_count=paper_row.get("page_count", len(pages)),
        pages=sorted(pages.values(), key=lambda p: p.number),
    )

    strategies = {
        "section_aware": SectionAwareChunker(),
        "fixed_size": FixedSizeChunker(chunk_tokens=500, overlap_tokens=80),
    }
    children: dict[str, list[Chunk]] = {}
    for s, ch in strategies.items():
        all_chunks = ch.chunk(paper)
        children[s] = [c for c in all_chunks if c.kind == "child"]

    bm25_by_strategy = {
        s: BM25Index().build([(c.id, c.text) for c in children[s]]) for s in strategies
    }

    # Precompute normalized embedding matrices once per strategy.
    mats: dict[str, tuple] = {}
    for s, ch in children.items():
        vecs = ctx.embedder.embed_documents([c.text for c in ch])
        mat = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mats[s] = (mat / norms, ch)

    def dense_search_fn(strategy: str, query: str, k: int):
        mat, ch = mats[strategy]
        qn = np.array(ctx.embedder.embed_query(query), dtype=np.float32)
        n = np.linalg.norm(qn) or 1.0
        scored = mat @ (qn / n)
        order = np.argsort(-scored)
        return [(ch[i].id, float(scored[i])) for i in order[:k]]

    exp = run_retrieval_experiment(
        ctx.settings,
        ctx.db,
        name=f"retrieval-{paper_id[-6:]}",
        paper_id=paper_id,
        dataset=ds,
        chunks_by_strategy=children,
        bm25_by_strategy=bm25_by_strategy,
        dense_search_fn=dense_search_fn,
        reranker=ctx.reranker,
    )
    return {"experiment_id": exp["id"], "rows": exp["results"]["rows"], "experiment": exp}
