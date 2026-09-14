import { useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import { useAsync, fmtMs } from "../hooks";
import {
  Breadcrumbs,
  Button,
  Card,
  Empty,
  ErrorNote,
  PageHeader,
  Stat,
} from "../components/ui";
import type { RetrievalResult, RetrievedChunk } from "../types";

const EXAMPLES = [
  "learning rate and optimizer",
  "training data and corpus size",
  "evaluation metrics on Natural Questions",
  "model size and parameters",
];

export default function Retrieve() {
  const { paperId = "" } = useParams();
  const paper = useAsync(() => api.getPaper(paperId), [paperId]);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<RetrievalResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  const run = async (q?: string) => {
    const text = (q ?? query).trim();
    if (text.length < 2) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.retrieve(paperId, text));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Breadcrumbs
        items={[
          { label: "Papers", to: "/papers" },
          { label: paper.data?.title || paperId, to: `/papers/${paperId}` },
          { label: "Retrieve" },
        ]}
      />
      <PageHeader
        kicker="Hybrid retrieval"
        title="Retrieval pipeline"
        sub="dense (ONNX) + sparse (BM25) → RRF fusion → cross-encoder rerank. Every stage's candidates and timings are shown."
      />

      <Card className="mb-6 animate-rise">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void run();
          }}
          className="flex gap-3"
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Query the paper…"
            className="flex-1 bg-transparent text-base outline-none placeholder:text-ink-faint"
          />
          <Button type="submit" busy={busy} disabled={query.trim().length < 2}>
            Retrieve
          </Button>
        </form>
        <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-line">
          {EXAMPLES.map((s) => (
            <button
              key={s}
              onClick={() => {
                setQuery(s);
                void run(s);
              }}
              className="px-3 py-1.5 text-xs border border-line rounded-sm text-ink-soft hover:bg-panel hover:border-accent/40 transition-colors cursor-pointer"
            >
              {s}
            </button>
          ))}
        </div>
      </Card>

      {error && <ErrorNote message={error} />}

      {result && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 mb-6 animate-rise">
            <Card><Stat label="Dense" value={result.dense_candidates} sub={fmtMs(result.stages.dense_ms)} /></Card>
            <Card><Stat label="Sparse (BM25)" value={result.sparse_candidates} sub={fmtMs(result.stages.sparse_ms)} /></Card>
            <Card><Stat label="Fused (RRF)" value={result.fused_candidates} sub={fmtMs(result.stages.fusion_ms)} /></Card>
            <Card><Stat label="Reranked" value={result.reranked_candidates} sub={fmtMs(result.stages.rerank_ms)} /></Card>
            <Card><Stat label="Final" value={result.final_count} sub={`${fmtMs(result.stages.total_ms)} total`} /></Card>
          </div>

          {result.evidence.length === 0 ? (
            <Empty title="No candidates" hint="Both retrieval paths returned nothing for this query." />
          ) : (
            <Card pad={false} className="animate-rise">
              <ul className="divide-y divide-line/70">
                {result.evidence.map((rc, i) => {
                  const c = rc.chunk;
                  const open = expanded === c.id;
                  return (
                    <li key={c.id}>
                      <button
                        onClick={() => setExpanded(open ? null : c.id)}
                        className="w-full text-left px-5 py-3 hover:bg-panel/50 transition-colors cursor-pointer"
                      >
                        <div className="flex items-center gap-3">
                          <span className="font-mono text-xs text-ink-faint w-6">{i + 1}</span>
                          <span className="font-mono text-xs text-accent w-14 shrink-0">
                            {c.page_start === c.page_end ? `p.${c.page_start}` : `p.${c.page_start}–${c.page_end}`}
                          </span>
                          <span className="text-[13px] text-ink-soft line-clamp-1 flex-1">
                            {c.text}
                          </span>
                          <Scores r={rc} />
                        </div>
                      </button>
                      {open && (
                        <div className="px-5 pb-4 pl-[3.75rem] animate-fadein">
                          <div className="flex flex-wrap gap-2 mb-3 text-[11px] text-ink-faint">
                            <span className="font-mono">{c.id}</span>
                            <span>·</span>
                            <span>{c.section_path.join(" › ") || "no section"}</span>
                            <span>·</span>
                            <span>{c.token_count} tokens · {c.chunk_type}</span>
                          </div>
                          <p className="text-sm leading-relaxed whitespace-pre-wrap max-h-72 overflow-y-auto text-ink-soft">
                            {c.text}
                          </p>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </Card>
          )}
        </>
      )}

      {!result && !error && !busy && (
        <Empty
          title="Run a query to inspect the pipeline"
          hint="Each result row shows its rank and score at every stage: dense, sparse, RRF, and rerank."
        />
      )}
    </>
  );
}

function Scores({ r }: { r: RetrievedChunk }) {
  const cell = (label: string, score: number | null, rank: number | null) => (
    <span className="text-[10px] font-mono text-ink-faint tabular-nums whitespace-nowrap">
      {label} {score != null ? score.toFixed(3) : "·"}
      {rank != null ? ` (#${rank})` : ""}
    </span>
  );
  return (
    <span className="flex items-center gap-3 shrink-0">
      {cell("d", r.dense_score, r.dense_rank)}
      {cell("s", r.sparse_score, r.sparse_rank)}
      {cell("rrf", r.rrf_score, r.rrf_rank)}
      {cell("rr", r.rerank_score, r.rerank_rank)}
    </span>
  );
}
