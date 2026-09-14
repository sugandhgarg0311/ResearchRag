# Research RAG

Evidence-first platform for understanding research papers and turning them into working code — fully local, no cloud accounts required.

Given a PDF, Research RAG parses it into a structured document (pages, sections, paragraphs, tables, formulas, figures — each with provenance), builds hierarchical semantic chunks, indexes them with hybrid dense + sparse (BM25) retrieval fused by Reciprocal Rank Fusion, reranks with a cross-encoder, and assembles a token-budgeted evidence package that grounds answers with page-level citations. It then derives an implementation specification in which every requirement carries a provenance tag (`EXPLICIT` / `INFERRED` / `EXTERNAL` / `UNKNOWN`) plus a verbatim source quote, generates a complete runnable project from that spec, executes it in a resource-limited sandbox, and compares produced metrics against the paper's reported values — reporting gaps as gaps instead of inventing numbers.

```
PDF ─► parse ─► chunk ─► index ─► retrieve ─► fuse ─► rerank ─► assemble ─► answer
                                                        │                        │
                                                        │                        ▼
                                                        │                   implementation spec
                                                        │                        │
                                                        │                        ▼
                                                        │                   generated project
                                                        │                        │
                                                        └──────────────────► sandbox run ─► metrics ─► reproduction report
                                                                                 │
                                                                                 └──────► evaluation (Recall@K, MRR, NDCG,
                                                                                          groundedness, citation correctness)
```

Default stack runs offline on a laptop: SQLite + embedded Qdrant + ONNX models (fastembed), PyMuPDF parsing, pure-Python BM25, extractive answering. LLM calls (OpenAI-compatible / Anthropic) and hosted rerankers are optional enhancements, never requirements.

---

## 1. Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 (tested 3.11) | `python -m venv .venv` |
| Node.js | ≥ 18 | Only for `frontend/` development (`npm run dev`); production UI is served by the API from `frontend/dist` |
| OS | Linux / macOS / Windows | Memory caps in the sandbox use `resource.setrlimit` on POSIX; on Windows they degrade to timeout + env scrubbing |

No PyTorch, no LangChain/LlamaIndex, no vector-DB server, no API key in the default path.

## 2. Install and quickstart

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"        # Windows: .venv\Scripts\pip install -e ".[dev]"

.venv/bin/researchrag ingest data/samples/rag_paper.pdf
.venv/bin/researchrag ask paper_16c17d9569a2 "What is RAG?"
.venv/bin/researchrag serve               # API :8000, OpenAPI at /docs

cd frontend && npm install && npm run dev # UI :5173, proxies /api → :8000
```

Production UI without a second process:

```bash
cd frontend && npm install && npm run build
cd .. && .venv/bin/researchrag serve      # built UI served at /
```

Copy `.env.example` to `.env` to override any setting (all `RESEARCHRAG_*`, §6). Runtime state lives under `data/` (git-ignored); `rm -rf data/` resets everything.

## 3. Repository layout

```
researchrag/          Python package — the entire backend (API, CLI, pipeline)
frontend/             Vite + React 18 + TypeScript + Tailwind web UI
tests/                pytest suite (114 tests, fully offline)
evals/                QA datasets for retrieval/answer evaluation
data/samples/         bundled sample paper (RAG, 19 pages) for first ingest
scripts/              mock_llm_server.py — offline OpenAI-compatible LLM stub
pyproject.toml        package metadata, dependencies, entry points, pytest/ruff config
.env.example          annotated catalogue of every RESEARCHRAG_* setting
```

### 3.1 `researchrag/` — backend package

**Composition root and configuration**

| File | Role |
|---|---|
| `__init__.py` | Package version (`__version__ = "1.0.0"`) |
| `config.py` | `Settings` (pydantic-settings, `RESEARCHRAG_` prefix, `.env` support); `get_settings()` (lru-cached); `masked_dict()` for safe `/api/health` exposure with API keys redacted |
| `app.py` | `AppContext` composition root: builds and caches Database, QdrantStore, parser, chunker, embedder, reranker, LLM, ingestion pipeline; per-paper retrievers/answering pipelines built on demand. Model weights load lazily so the API boots fast |
| `cli.py` | `researchrag` entry point (12 subcommands, §7); `--json` output for scripting |
| `tokens.py` | `get_token_counter()` — tiktoken `cl100k` when available, chars/4 heuristic otherwise; single choke point for all budgets (chunker, assembler) |
| `evidence.py` | `EvidenceAssembler` — dedup by chunk id, per-section parent-context header (once per section), 4000-token global budget with drop-context-then-truncate degradation and explicit `truncated` flag, per-item page/page-range citations |

**`api/` — FastAPI application**

| File | Role |
|---|---|
| `api/app.py` | `create_app()` factory; mounts the six routers under `/api`, exposes `/healthz` + rich `/api/health` (embedder/reranker/LLM status, config, counts); serves `frontend/dist` at `/` in production with SPA-deep-link fallback and API-404 preservation; OpenAPI at `/docs` |
| `api/deps.py` | Request-scoped `AppContext`, `require_paper` existence guard, `create_job`/`run_job` async-job helper (202 + `job_id`, daemon thread, signature-gated progress callbacks, results in kv as `job_result:{id}`), structured `{"error": {"code", "message"}}` errors (never tracebacks) |
| `api/routes/papers.py` | Paper CRUD, upload (PDF → ingest job), `structure`, `chunks`, page/block image rendering, runs listing, reproduction records |
| `api/routes/search.py` | `POST /papers/{id}/chat` (grounded answer) and `/retrieve` (full hybrid pipeline with per-stage timings) |
| `api/routes/specs.py` | Spec generation (202 job), spec get with requirements join, per-requirement status transitions |
| `api/routes/implementations.py` | Implementation generation (202 job), get, file content, sandboxed runs (202 job) |
| `api/routes/evaluation.py` | Evaluation/experiment dataset listing, evaluation runs, experiment sweeps (202 jobs) |
| `api/routes/jobs.py` | Job polling (`GET /api/jobs/{id}`) and listing (global + per-paper) |

**`answering/` — grounded question answering**

| File | Role |
|---|---|
| `answering/pipeline.py` | `AnsweringPipeline`: retrieve evidence → LLM path (expects trailing `CONFIDENCE:`/`SUFFICIENCY:` markers) or `ExtractiveAnswerer` fallback on any LLM failure; uniform `Answer` model (`mode` = `llm`/`extractive`) |
| `answering/prompts.py` | SYSTEM/USER prompt contracts (`[n] (p.3, section) text` evidence rendering), `strip_markers` marker parser (unparseable → `confidence=unknown`) |

**`parsing/` — PDF → structured document**

| File | Role |
|---|---|
| `parsing/base.py` | `BaseParser` interface (`parse() → Paper`) — the approved extension point for new backends |
| `parsing/pymupdf_parser.py` | Default backend: fast (~seconds), no torch; emits `Paper → Page → Block` with stable ids `{paper}:{page:03d}:{order:04d}`, heading levels/section paths, table+caption atomicity; paper id = `paper_` + 12 hex of file sha256 (idempotent re-ingest) |
| `parsing/docling_parser.py` | Optional high-fidelity backend (`RESEARCHRAG_PARSER_BACKEND=docling`, extra `researchrag[docling]`); same model output |

**`chunking/` — section-aware hierarchical chunking**

| File | Role |
|---|---|
| `chunking/base.py` | `BaseChunker` interface |
| `chunking/section_aware.py` | Default: group blocks into sections → emit children (≤400 tokens, table+caption atomic) → emit parents (≤3000 tokens, every child references an existing parent); deterministic ids `{paper}:{child\|parent}:{ordinal}:{sha1(text)[:12]}`; cross-page chunks keep a page *range* — pages are provenance metadata, never split boundaries |
| `chunking/fixed_size.py` | Experimental baseline (500/80) so the evaluation framework can quantify what section-awareness buys |
| `chunking/sentences.py` | Sentence splitter shared by the chunk fitter and the extractive answerer |

**`embeddings/` — dense vectors**

| File | Role |
|---|---|
| `embeddings/base.py` | `BaseEmbedder` interface |
| `embeddings/factory.py` | `make_embedder()` — fastembed first, automatic fallback to hashing on download failure |
| `embeddings/fastembed_embedder.py` | ONNX `BAAI/bge-base-en-v1.5` (768-d) via ONNX Runtime — no PyTorch |
| `embeddings/hash_embedder.py` | Deterministic char-n-gram hashing into 768 L2-normalized dims; zero weights; keeps the system fully offline |

**`retrieval/` — hybrid search**

| File | Role |
|---|---|
| `retrieval/dense.py` | Qdrant cosine search (top-50) over the embedded collection |
| `retrieval/sparse.py` | Pure-Python `BM25Index` (`k1=1.5, b=0.75`, `[a-z0-9][a-z0-9._\-]*` tokenizer so `top-k`, `3e-5`, `batch_size` stay whole); per-paper JSON persistence with int-key round-trip |
| `retrieval/rrf.py` | `reciprocal_rank_fusion` — `score(d) = Σ wᵢ/(k+rankᵢ)`, default `k=60`; rank-based so cosine and BM25 scores need no calibration |
| `retrieval/hybrid.py` | `HybridRetriever.retrieve` — single entry point returning candidates plus per-stage timings at every stage |
| `retrieval/rerank.py` | `LocalCrossEncoderReranker` (ONNX `Xenova/ms-marco-MiniLM-L-6-v2`, fused top-30 → final top-8), `HostedReranker` (Cohere/Jina), `NullReranker` (`RESEARCHRAG_RERANKER=none`) |

**`spec/` — paper → implementation specification (provenance before code)**

| File | Role |
|---|---|
| `spec/generator.py` | `SpecGenerator`: LLM path (JSON with per-requirement `EXPLICIT/INFERRED/EXTERNAL/UNKNOWN` + verbatim `source_quote`) or deterministic rule path; persists spec row + one row per requirement |
| `spec/extractor_rules.py` | Offline regex rule set (learning rate, optimizer, batch size, hidden size/layers, hardware, top-k, corpus size, sequence length, …); hits become `EXPLICIT` with quote + page + chunk id; misses surface as `open_questions`, never invented |
| `spec/evidence.py` | Shared (chunk id, text, section, page, kind) fetcher used by both the rule extractor and the spec API endpoint |
| `spec/loader.py` | Spec/requirement read helpers for codegen and the API |

**`codegen/` — spec → runnable project**

| File | Role |
|---|---|
| `codegen/scaffold.py` | Deterministic complete project: `config.yaml` (values only from `EXPLICIT` requirements; gaps → `null` + `# not specified in paper`), `run_smoke.sh`, `tests/test_smoke.py`, `src/<slug>/{model,data,train,evaluate,infer}.py`, `README.md`, `requirements.txt`; codegen consumes only non-rejected requirements |
| `codegen/generator.py` | `generate_and_validate`: optional LLM file enhancement (unparseable output discarded, scaffold kept as the valid floor) → write to `data/artifacts/implementations/{id}/` → persist + `validate_project` |
| `codegen/validator.py` | Three reported stages: syntax (AST parse of every file) → imports (intra-project resolution) → tests + smoke run (pass judged on combined stdout+stderr, since unittest reports `OK` on stderr) |

**`execution/`, `repro/`, `evaluation/`, `llm/`, `models/`, `storage/`**

| File | Role |
|---|---|
| `execution/runner.py` | `run_command`: wall-clock timeout (default 120 s, SIGKILL), POSIX memory cap, scrubbed child env (`PYTHONUTF8=1`, UTF-8 I/O, no `*_API_KEY`/tokens propagate), UTF-8-decoded capture, optional log file; generated code runs only through here |
| `repro/metrics.py` | `parse_metrics_from_output` (the sole metrics channel: `METRICS: {...}` stdout lines) and `record_run_results` (run − paper difference; missing paper value → `null` + "paper does not report this metric") |
| `evaluation/dataset.py` | `load_dataset` / `resolve_gold_chunks` — gold references may be chunk ids, section paths, or `AUTO:<text fragment>` resolved by text match (chunking-change-proof) |
| `evaluation/metrics.py` | Pure functions: `recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k`, `context_precision/recall_at_k`, `citation_correctness`, `groundedness` (stopword-filtered term overlap), `answer_relevance` |
| `evaluation/runner.py` | `evaluate_retrieval` (pluggable `retrieve_fn` so strategies swap without touching the pipeline) and `evaluate_answers` |
| `evaluation/service.py` | `run_evaluation` / `run_experiments` (chunking × retrieval-mode sweeps) backing the API/CLI jobs |
| `evaluation/experiments.py` | Experiment matrix construction and aggregation |
| `llm/base.py`, `llm/factory.py` | Provider interface + constructor (`openai_compatible` / `anthropic` / `offline`) |
| `llm/openai_compatible.py` | OpenAI-compatible chat client with timeout + retries (covers OpenAI, Ollama, vLLM, LM Studio, the mock server) |
| `llm/anthropic.py` | Anthropic Messages API client |
| `llm/extractive.py` | `ExtractiveAnswerer` — stopword-filtered sentence/overlap ranking over evidence, verbatim cited composition, ≤0.05 overlap → explicit "paper does not contain" with `sufficiency=insufficient` |
| `models/document.py` | `Paper`, `Page`, `Block` (stable ids, heading levels + section paths, table captions attached to table blocks) |
| `models/chunk.py` | `Chunk` (`make_chunk_id`, `pages_display` rendering `p.3` / `p.3–5`) |
| `models/retrieval.py` | `RetrievedChunk`, `RetrievalResult` (per-stage timings), `Answer`, `SpecRequirement` (`SourceKind`, review `status`), `Citation` |
| `storage/database.py` | SQLite schema + accessors: papers/blocks/chunks, specs + requirements (ids minted per requirement), implementations + files, runs, repro records, jobs, generic `kv_get/kv_set` (job results, paper metric values) |
| `storage/qdrant_store.py` | Embedded-Qdrant wrapper (`qid()` chunk→ULID mapping, upsert/delete/search); single-process file lock — server *or* CLI holds `data/qdrant`, never both |
| `ingestion/pipeline.py` | `IngestionPipeline.ingest_file`: parse → chunk → embed → Qdrant + BM25 JSON + SQLite + provenance index, with progress callbacks |

### 3.2 `frontend/` — web UI (Vite 5 + React 18 + TypeScript 5 + Tailwind 3)

`package.json` scripts: `dev` (Vite :5173, `/api` proxied to :8000), `build` (`tsc -b && vite build` → `dist/`, served by the API), `preview`.

| File | Role |
|---|---|
| `index.html` | Vite entry HTML (`#root` mount) |
| `vite.config.ts` | React plugin, dev proxy `/api → http://localhost:8000` |
| `tailwind.config.js` | Theme tokens bound to CSS variables (`paper/panel/card/ink/line/accent/ok/warn/danger/infer/external`), `display/sans/mono/math` stacks, `content` width, `card/lift` shadows, `rise/fadein/shimmer` keyframes |
| `postcss.config.js` | Tailwind + autoprefixer pipeline |
| `tsconfig.json` / `tsconfig.node.json` | App + node build configs |
| `src/main.tsx` | React root: `BrowserRouter → ThemeProvider → ViewModeProvider → App` |
| `src/App.tsx` | Route table (all routes render inside `Layout`) |
| `src/api.ts` | Typed fetch client (relative URLs work against both the dev proxy and production serving); `ApiError`; papers/structure/chunks/upload/delete, chat/retrieve, specs + requirement status, implementations/files/runs, reproduction, evaluations/experiments, jobs |
| `src/types.ts` | TypeScript mirrors of the API models (`Paper`, `Job`, `PaperStructure`, `Chunk`, `RetrievalResult`, `Answer`, `Citation`, `Spec`, `SpecRequirement`, `Implementation`, `RunRecord`, `ReproductionRecord`, `EvalRun`, `Experiment`, `Health`) |
| `src/hooks.ts` | `useAsync` (fetch + loading/error/reload), `useJob` (1.2 s job polling to terminal state), `fmtMs/fmtDate/fmtNum` |
| `src/view.tsx` | Simplified/Advanced view-mode context (localStorage `rr-view-mode`); simplified hides engine internals, advanced reveals configs, raw JSON, chunk ids |
| `src/theme.tsx` | Light/dark theme context (localStorage `rr-theme`, OS default, `dark` class on `<html>`) + sun/moon toggle |
| `src/index.css` | CSS variables for both themes, base styles, `prose-chunk`, math font stack (STIX/XITS/Cambria), `stagger` entrance, `skeleton` shimmer, `prefers-reduced-motion` kill-switch |
| `src/components/Layout.tsx` | Sidebar (top-bar on mobile) with Dashboard/Papers/Chat/Jobs/System nav, view toggle, embedder/reranker/LLM status footer |
| `src/components/ui.tsx` | Design-system primitives: `Card`, `PageHeader`, `Button`, `Badge` (+ `sourceTone` provenance colors), `StatusDot`, `Skeleton`, `Stagger`, `Empty`, `ErrorNote`, `Loading`, `Table`/`Td`, `Tabs`, `Stat`, `MetricBar`, `Code`, `Prose`, `Breadcrumbs` |
| `src/pages/Dashboard.tsx` | Overview: papers/processing counts, engine status cards, paper grid |
| `src/pages/Papers.tsx` | Library table + PDF upload with ingest-job progress bar + delete |
| `src/pages/Chat.tsx` | Global grounded chat over any ready paper, suggestions, citations, raw JSON (advanced) |
| `src/pages/PaperDetail.tsx` | Per-paper workspace: Overview (pipeline-action links, metadata, jobs), Structure (section tree + figure thumbnails), Chunks (searchable table with formula rendering), Pages (rendered page images with preloading + retry) |
| `src/pages/Ask.tsx` | Per-paper QA with suggestions, confidence/sufficiency badges, stage timings, retrieval tuning (advanced) |
| `src/pages/Retrieve.tsx` | Hybrid-pipeline debugger: per-stage candidate counts + timings, per-hit dense/sparse/RRF/rerank ranks and scores, expandable chunk text |
| `src/pages/Specs.tsx` | Spec list + async generation with job progress |
| `src/pages/SpecDetail.tsx` | Requirement review: EXPLICIT/INFERRED filters, verbatim quotes with pages, open/accepted/rejected/implemented workflow (codegen consumes non-rejected) |
| `src/pages/Implementations.tsx` | Project list + generation from a chosen spec with validation summary |
| `src/pages/ImplementationDetail.tsx` | Validation stages (syntax/imports/tests/smoke with pass/fail badges), file browser + viewer, sandbox run buttons (smoke/tests/train), run history |
| `src/pages/Reproduction.tsx` | Paper-vs-run metric table (gaps shown as gaps) + manual paper-value entry form |
| `src/pages/Evaluations.tsx` | Eval runs with retrieval/answer metric bars + per-question detail |
| `src/pages/Experiments.tsx` | Chunking × retrieval-mode sweep table with per-group best-recall highlighting |
| `src/pages/Jobs.tsx` | Background-job monitor (ingest/spec/implement/run/evaluate/experiment) with progress, auto-refresh while active |
| `src/pages/Health.tsx` | Pipeline-component status (embedder/reranker/LLM/parser), fallback explanation, full configuration (advanced, copyable) |

### 3.3 `tests/` — pytest suite (114 tests, offline, ~2 s)

| File | Covers |
|---|---|
| `test_rrf.py` | RRF formula exactness, empty lists, `k<=0`/weight-mismatch `ValueError`, intra-/inter-list duplicates, constructed ties, zero weights, 1000-doc monotonicity |
| `test_bm25.py` | Tokenizer (`top-k`, `3e-5` stay whole), ranking, persistence round-trip incl. integer posting keys |
| `test_chunking.py` | Token budgets, no orphan parents, atomic table+caption, cross-page ranges, deterministic ids |
| `test_evidence.py` | Parent headers once per section, global budget + truncation flag, dedup, citation page ranges |
| `test_metrics.py` | recall/precision/MRR/NDCG, context precision/recall, citation correctness, groundedness, relevance |
| `test_models.py` | Id stability, provenance fields |
| `test_embedders.py` | Hashing embedder determinism, normalization, dim |
| `test_extractive.py` | Sentence ranking, verbatim composition, insufficient-overlap behavior |
| `test_spec_rules.py` | Rule-extractor hits, EXPLICIT provenance, open questions for gaps |
| `test_repro.py` | `METRICS` parsing, run-vs-paper comparison incl. missing paper values |
| `test_formulas.py` | Formula/equation block handling through chunking |
| `test_codegen_portability.py` | Windows/no-bash fallbacks, UTF-8 scrubbing/decoding, scaffold template invariants, unittest-stderr pass detection |

### 3.4 Data, evals, scripts, config files

| Path | Role |
|---|---|
| `evals/rag_eval_lwis2020.json` | 32 QA pairs over the sample paper; gold chunks as ids/sections/`AUTO:` fragments (100% resolvable post-ingest) |
| `data/samples/rag_paper.pdf` | Bundled sample paper (RAG, 19 pages, ~0.9 MB) — the only tracked file under `data/`; everything else there is git-ignored runtime state |
| `scripts/mock_llm_server.py` | Offline OpenAI-compatible LLM stub (`:11435`): composes verbatim cited sentences from evidence so the full generative path (prompt → markers → parsing) is exercisable without weights or keys |
| `pyproject.toml` | `researchrag` 1.0.0, MIT, `requires-python>=3.10`; runtime deps (fastapi, uvicorn, pydantic, PyMuPDF, Pillow, fastembed, qdrant-client, tiktoken); `docling`/`dev` extras; `researchrag` console script; pytest + ruff config |
| `.env.example` | Every `RESEARCHRAG_*` variable with defaults and commentary (§6) |
| `.gitignore` | Excludes `.venv`, `__pycache__`, `.env`, all of `data/` except `data/samples/`, `frontend/node_modules`, `frontend/dist`, build info, caches, egg-info |

## 4. Storage model

| Store | Contents |
|---|---|
| SQLite `data/researchrag.db` | papers, blocks, chunks, specs + requirements, implementations + files, runs, repro records, jobs, kv (job results, paper metric values). Source of truth for document structure and experiment records |
| Embedded Qdrant `data/qdrant` | Dense vectors (`researchrag_chunks`, cosine) keyed by chunk id. Exclusive file lock: one process (server *or* CLI), never both |
| `data/index/bm25_{paper}.json` | Per-paper BM25 index (df, postings, vocab) — inspectable, deletable per paper |
| `data/artifacts/implementations/{id}/` | Generated projects + run logs |
| `data/raw/`, `data/uploads/` | Immutable input PDFs |

## 5. Quickstart flows

```bash
.venv/bin/researchrag papers
.venv/bin/researchrag structure paper_16c17d9569a2
.venv/bin/researchrag retrieve paper_16c17d9569a2 "learning rate" --k 8
.venv/bin/researchrag spec paper_16c17d9569a2
.venv/bin/researchrag implement paper_16c17d9569a2
.venv/bin/researchrag run impl_XXXXXXXX --kind smoke
.venv/bin/researchrag evaluate paper_16c17d9569a2 --k 10
.venv/bin/researchrag experiment paper_16c17d9569a2
```

With an LLM (optional): point `RESEARCHRAG_LLM_PROVIDER=openai_compatible` at Ollama (`:11434`), any OpenAI-compatible endpoint, or the bundled mock (`python scripts/mock_llm_server.py`, `:11435`).

## 6. Configuration

All settings are `RESEARCHRAG_*` env vars (or `.env` keys); see `.env.example`. Defaults form the fully-offline zero-key system:

- Parsing `pymupdf` · chunking `section_aware` (child 400 / parent 3000 tokens) · embeddings `BAAI/bge-base-en-v1.5` with hashing fallback
- Retrieval dense/sparse top-50 · RRF `k=60`, weights 1.0/1.0 · reranker local MiniLM top-30→8 · evidence budget 4000 tokens with parent context
- LLM `offline` · sandbox 120 s / 1024 MB / no network

## 7. CLI reference (`researchrag --help`)

| Command | Effect |
|---|---|
| `serve` | Start FastAPI (:8000) + serve built UI; `/docs` for OpenAPI |
| `ingest <pdf>` | Parse → chunk → index a paper (idempotent by file hash) |
| `papers` / `remove <id>` | List library / delete paper and its indexes |
| `structure <id>` | Section tree |
| `retrieve <id> "query"` | Hybrid pipeline with per-stage scores |
| `ask <id> "question"` | Grounded answer with citations |
| `spec <id>` | Implementation spec (rule extractor offline, LLM path if configured) |
| `implement <id>` | Generate + validate project |
| `run <impl> --kind smoke\|tests\|train` | Sandboxed execution |
| `evaluate <id>` / `experiment <id>` | Score retrieval+answers / sweep chunking × retrieval modes |

## 8. REST API

Routers: `papers`, `search` (chat/retrieve), `specs`, `implementations`, `evaluation`, `jobs` — all under `/api`, plus `/healthz` and `/api/health`. Long work is always `202 + {job_id}` (ingest, spec, implement, run, evaluate, experiment); poll `GET /api/jobs/{id}` or `/api/papers/{id}/jobs`. Errors are structured JSON; unknown ids → 404, bad input → 400.

## 9. Evaluation

`evals/rag_eval_lwis2020.json` (32 questions) drives `evaluate` (recall@K, precision@K, MRR, NDCG@K, context precision/recall, citation correctness, groundedness, answer relevance) and `experiment` (section_aware × fixed_size, dense × sparse × hybrid). Reference numbers are recorded in the codebase history; re-run on your machine for current figures.

## 10. Invariants

1. Provenance is never dropped (chunk block ids + page ranges, cited answer sentences, quoted EXPLICIT requirements — or explicit `unknown`).
2. Pages are metadata, never chunk boundaries.
3. Missing information is reported (open questions, `paper_value: null`, `null` config with comment, `insufficient` answers) — never fabricated.
4. Generated code runs only in the sandbox.
5. RRF stays rank-based and configurable; retrieval changes go through the experiment framework.
6. Same paper + same config → same chunk ids, BM25 index, rule-path spec (tests enforce determinism).

## License

MIT
