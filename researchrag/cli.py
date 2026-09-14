"""Research RAG command-line interface.

Every server workflow has a CLI equivalent, so the platform is usable
without a browser and scriptable from the shell:

    researchrag ingest path/to/paper.pdf
    researchrag ask paper_... "question"
    researchrag retrieve paper_... "query"
    researchrag evaluate paper_...
    researchrag spec paper_...
    researchrag implement paper_...
    researchrag run impl_...
    researchrag serve

The CLI talks to the same AppContext the API uses, so results are
identical. Long operations print progress; add --json for machine output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from researchrag import __version__

# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------

def _print(data, as_json: bool, human: str) -> None:
    if as_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(human)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_serve(args) -> int:
    import uvicorn

    _setup_logging(args.verbose)
    print(f"Research RAG API on http://{args.host}:{args.port} (docs: /docs)")
    uvicorn.run(
        "researchrag.api.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def cmd_ingest(args) -> int:
    from researchrag.app import get_app_context

    _setup_logging(args.verbose)
    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1
    ctx = get_app_context()

    def progress(p: float, detail: str) -> None:
        if not args.json:
            print(f"  {p * 100:3.0f}% {detail}")

    result = ctx.ingestion.ingest_file(path, progress=progress, paper_id=args.paper_id)
    if args.json:
        _print(result.__dict__, True, "")
    else:
        print(
            f"ingested {result.paper_id}: {result.page_count} pages, "
            f"{result.chunk_count} child chunks ({result.parent_count} parents) "
            f"in {result.timings.get('total_s')}s"
        )
    return 0


def cmd_papers(args) -> int:
    from researchrag.app import get_app_context

    _setup_logging(args.verbose)
    papers = get_app_context().db.list_papers()
    if args.json:
        _print({"papers": papers}, True, "")
        return 0
    if not papers:
        print("no papers ingested yet (use: researchrag ingest <file.pdf>)")
        return 0
    for p in papers:
        title = (p.get("title") or p.get("filename") or "?")[:60]
        print(f"{p['id']}  {p['status']:<10} {p.get('page_count', 0):>3}p  {title}")
    return 0


def cmd_remove(args) -> int:
    from researchrag.app import get_app_context

    _setup_logging(args.verbose)
    ctx = get_app_context()
    ctx.ingestion.remove_paper(args.paper_id)
    ctx.invalidate_retriever(args.paper_id)
    print(f"removed {args.paper_id}")
    return 0


def cmd_ask(args) -> int:
    from researchrag.app import get_app_context

    _setup_logging(args.verbose)
    ctx = get_app_context()
    answerer = ctx.get_answerer(args.paper_id, use_llm=not args.no_llm)
    ans = answerer.answer(args.question, args.paper_id, config=None)
    if args.json:
        _print(ans.model_dump(), True, "")
        return 0
    print(f"Q: {ans.question}")
    print(f"A ({ans.mode}, confidence={ans.confidence}, sufficiency={ans.sufficiency}):")
    print(ans.answer)
    print("\nSources:")
    for c in ans.citations:
        print(f"  - {c.section} ({c.page_range})")
    return 0


def cmd_retrieve(args) -> int:
    from researchrag.app import get_app_context
    from researchrag.models.retrieval import RetrievalConfig

    _setup_logging(args.verbose)
    ctx = get_app_context()
    retriever = ctx.get_retriever(args.paper_id)
    config = RetrievalConfig(final_top_k=args.k)
    result = retriever.retrieve(args.query, args.paper_id, config=config)
    if args.json:
        _print(result.model_dump(), True, "")
        return 0
    print(f"query: {result.query}")
    print(
        f"candidates: dense={result.dense_candidates} sparse={result.sparse_candidates} "
        f"fused={result.fused_candidates} reranked={result.reranked_candidates} "
        f"final={result.final_count}"
    )
    for i, rc in enumerate(result.evidence, 1):
        c = rc.chunk
        score = rc.rerank_score if rc.rerank_score is not None else rc.rrf_score
        print(f"\n[{i}] {c.section_display} ({c.pages_display}) score={score:.4f}")
        print(f"    {c.text[:200].replace(chr(10), ' ')}")
    return 0


def cmd_structure(args) -> int:
    from researchrag.app import get_app_context

    _setup_logging(args.verbose)
    ctx = get_app_context()
    rows = ctx.db.get_blocks(args.paper_id)
    sections = [
        r for r in (dict(r) for r in rows) if r["block_type"] == "heading"
    ]
    if args.json:
        _print({"sections": sections}, True, "")
        return 0
    for s in sections:
        path = json.loads(s["section_path_json"]) if s.get("section_path_json") else []
        indent = "  " * (s.get("heading_level") or 1 - 1)
        print(f"{indent}- {path[-1] if path else s['text'][:60]} (p.{s['page']})")
    return 0


def cmd_evaluate(args) -> int:
    from researchrag.app import get_app_context
    from researchrag.evaluation import service

    _setup_logging(args.verbose)
    ctx = get_app_context()
    names = service.list_datasets(ctx.settings)
    if not names:
        print("error: no eval datasets found", file=sys.stderr)
        return 1
    name = args.dataset or names[0]
    if name not in names:
        print(f"error: unknown dataset '{name}'. Available: {', '.join(names)}", file=sys.stderr)
        return 1
    if not args.json:
        print(f"evaluating on {name} (k={args.k}) …")
    result = service.run_evaluation(ctx, args.paper_id, name, k=args.k)
    if args.json:
        _print(result, True, "")
        return 0
    s = result["summary"]
    print(f"\nretrieval (k={s['k']}):")
    for k, v in s["retrieval"].items():
        print(f"  {k:>18}: {v}")
    print("answers:")
    for k, v in s["answers"].items():
        print(f"  {k:>18}: {v}")
    print(f"\neval run: {result['eval_run_id']}")
    return 0


def cmd_experiment(args) -> int:
    from researchrag.app import get_app_context
    from researchrag.evaluation import service

    _setup_logging(args.verbose)
    ctx = get_app_context()
    names = service.list_datasets(ctx.settings)
    if not names:
        print("error: no eval datasets found", file=sys.stderr)
        return 1
    if not args.json:
        print(f"running retrieval experiment on {names[0]} …")
    result = service.run_experiments(ctx, args.paper_id, names[0])
    if args.json:
        _print(result, True, "")
        return 0
    print(f"\n{'strategy':<14} {'retrieval':<14} {'k':>3} {'recall':>7} {'mrr':>7} {'ndcg':>7}")
    for r in result["rows"]:
        if r["top_k"] == 10:
            print(
                f"{r['strategy']:<14} {r['retrieval']:<14} {r['top_k']:>3} "
                f"{r['recall@k']:>7} {r['mrr']:>7} {r['ndcg@k']:>7}"
            )
    print(f"\nexperiment: {result['experiment_id']}")
    return 0


def cmd_spec(args) -> int:
    from researchrag.app import get_app_context
    from researchrag.spec.evidence import spec_evidence
    from researchrag.spec.generator import SpecGenerator

    _setup_logging(args.verbose)
    ctx = get_app_context()
    paper = ctx.db.get_paper(args.paper_id)
    if paper is None:
        print(f"error: paper {args.paper_id} not found", file=sys.stderr)
        return 1
    generator = SpecGenerator(ctx.settings, ctx.llm)
    evidence = spec_evidence(ctx, args.paper_id, limit=args.max_evidence)
    spec = generator.generate(args.paper_id, paper.get("title"), evidence)
    ctx.db.insert_spec(spec)
    if args.json:
        _print(spec.model_dump(), True, "")
        return 0
    stats = spec.stats()
    print(f"spec: {spec.id}  ({stats['total']} requirements: {stats['by_source']})")
    for r in spec.requirements:
        page = f" p.{r.source_page}" if r.source_page else ""
        print(f"  {r.id} [{r.source.value}]{page} {r.requirement}")
    if spec.open_questions:
        print("\nopen questions:")
        for q in spec.open_questions:
            print(f"  - {q}")
    return 0


def cmd_implement(args) -> int:
    from researchrag.app import get_app_context
    from researchrag.codegen.generator import CodeGenerator
    from researchrag.spec.evidence import spec_evidence
    from researchrag.spec.loader import spec_from_db

    _setup_logging(args.verbose)
    ctx = get_app_context()
    if args.spec_id is None:
        specs = ctx.db.list_specs(args.paper_id)
        if not specs:
            print("error: no spec found; run `researchrag spec <paper_id>` first", file=sys.stderr)
            return 1
        args.spec_id = specs[0]["id"]
    spec = spec_from_db(ctx, args.spec_id)
    if spec is None:
        print(f"error: spec {args.spec_id} not found", file=sys.stderr)
        return 1
    generator = CodeGenerator(ctx.settings, ctx.db, llm=None if args.no_llm else ctx.llm)
    evidence = spec_evidence(ctx, args.paper_id)
    impl_id, validation, project_dir = generator.generate_and_validate(spec, evidence)
    if args.json:
        _print(
            {
                "implementation_id": impl_id,
                "validation": {
                    "ok": validation.ok,
                    "stages": [s.__dict__ for s in validation.stages],
                    "errors": validation.errors,
                },
                "project_dir": str(project_dir),
            },
            True,
            "",
        )
        return 0
    print(f"implementation: {impl_id}  (project: {project_dir})")
    for s in validation.stages:
        print(f"  {s.name:<10} {s.status:<8} {s.detail[:80]}")
    if validation.errors:
        print("  errors:")
        for e in validation.errors:
            print(f"    - {e[:120]}")
    return 0


def cmd_run(args) -> int:
    from researchrag.app import get_app_context
    from researchrag.execution.runner import run_command, shell_available
    from researchrag.repro.metrics import parse_metrics_from_output
    from researchrag.storage.database import new_id, now_iso

    _setup_logging(args.verbose)
    ctx = get_app_context()
    impl = ctx.db.get_implementation(args.impl_id)
    if impl is None:
        print(f"error: implementation {args.impl_id} not found", file=sys.stderr)
        return 1
    project_dir = ctx.settings.artifacts_root / "implementations" / args.impl_id
    for f in impl["files"]:
        target = project_dir / f["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f["content"], encoding="utf-8")
        if f["kind"] == "script":
            target.chmod(0o755)

    pkg = next((d for d in (project_dir / "src").iterdir() if d.is_dir()), None)
    module = pkg.name.replace("-", "_") if pkg else "project"
    if args.kind == "smoke":
        cmd = ["bash", "run_smoke.sh"]
        extra = {"PYTHON_BIN": sys.executable}
    elif args.kind == "tests":
        cmd = ["bash", "-c", "python tests/test_smoke.py"]
        extra = {"PYTHON_BIN": sys.executable}
    else:
        cmd = ["bash", "-c", f"python -m {module}.train --config config.yaml"]
        extra = {"PYTHON_BIN": sys.executable}
    if not shell_available():
        # Plain Windows without Git Bash: run the same steps with this
        # interpreter instead of failing with FileNotFoundError.
        if args.kind == "train":
            cmd = [sys.executable, "-m", f"{module}.train", "--config", "config.yaml"]
        else:
            cmd = [sys.executable, "tests/test_smoke.py"]
        extra = {}

    log_path = project_dir / "logs" / f"run_{new_id('run')[-6:]}.log"
    result = run_command(
        cmd,
        cwd=project_dir,
        timeout_s=ctx.settings.execution_timeout_s,
        memory_mb=ctx.settings.execution_memory_mb,
        log_path=log_path,
        extra_env=extra,
    )
    metrics = parse_metrics_from_output(result.stdout)
    run_record = {
        "id": new_id("run"),
        "paper_id": impl["paper_id"],
        "implementation_id": args.impl_id,
        "kind": args.kind,
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
    if args.json:
        _print(run_record, True, "")
        return 0
    print(f"run: {run_record['status']} in {run_record['duration_ms']}ms (log: {log_path})")
    if metrics:
        print(f"metrics: {json.dumps(metrics)}")
    if result.returncode != 0:
        print("stderr:", result.stderr[-400:])
    return 0 if result.returncode == 0 else 1


def cmd_version(_args) -> int:
    print(f"researchrag {__version__}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="researchrag",
        description="Research RAG — paper understanding, evidence retrieval, implementation reproduction.",
    )
    p.add_argument("--version", action="version", version=f"researchrag {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--json", action="store_true", help="machine-readable output")
        sp.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    sp = sub.add_parser("serve", help="run the API server")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("ingest", help="parse + chunk + index a PDF")
    sp.add_argument("path")
    sp.add_argument("--paper-id", default=None)
    common(sp)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("papers", help="list ingested papers")
    common(sp)
    sp.set_defaults(func=cmd_papers)

    sp = sub.add_parser("remove", help="delete a paper and all its indexes")
    sp.add_argument("paper_id")
    common(sp)
    sp.set_defaults(func=cmd_remove)

    sp = sub.add_parser("ask", help="grounded question answering")
    sp.add_argument("paper_id")
    sp.add_argument("question")
    sp.add_argument("--no-llm", action="store_true", help="force extractive answering")
    common(sp)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("retrieve", help="raw hybrid retrieval (debug)")
    sp.add_argument("paper_id")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=8)
    common(sp)
    sp.set_defaults(func=cmd_retrieve)

    sp = sub.add_parser("structure", help="print the section tree")
    sp.add_argument("paper_id")
    common(sp)
    sp.set_defaults(func=cmd_structure)

    sp = sub.add_parser("evaluate", help="run the eval dataset metrics")
    sp.add_argument("paper_id")
    sp.add_argument("--dataset", default=None)
    sp.add_argument("--k", type=int, default=10)
    common(sp)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("experiment", help="retrieval architecture experiment")
    sp.add_argument("paper_id")
    common(sp)
    sp.set_defaults(func=cmd_experiment)

    sp = sub.add_parser("spec", help="generate an implementation specification")
    sp.add_argument("paper_id")
    sp.add_argument("--max-evidence", type=int, default=60)
    common(sp)
    sp.set_defaults(func=cmd_spec)

    sp = sub.add_parser("implement", help="generate + validate an implementation project")
    sp.add_argument("paper_id")
    sp.add_argument("--spec-id", default=None)
    sp.add_argument("--no-llm", action="store_true", help="scaffold only (deterministic)")
    common(sp)
    sp.set_defaults(func=cmd_implement)

    sp = sub.add_parser("run", help="run a generated project (sandboxed)")
    sp.add_argument("impl_id")
    sp.add_argument("--kind", choices=["smoke", "tests", "train"], default="smoke")
    common(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("version", help="print version")
    sp.set_defaults(func=cmd_version)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
