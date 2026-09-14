import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { useAsync, fmtMs } from "../hooks";
import { useViewMode } from "../view";
import {
  Badge,
  Breadcrumbs,
  Button,
  Card,
  Code,
  ErrorNote,
  PageHeader,
  Skeleton,
  sourceTone,
} from "../components/ui";
import type { Answer, RetrievalConfig } from "../types";

const SUGGESTIONS = [
  "What model architecture does the paper use, and how large is it?",
  "What learning rate and optimizer were used for training?",
  "How was the evaluation dataset constructed?",
  "What hardware was used for training?",
];

export default function Ask() {
  const { paperId = "" } = useParams();
  const paper = useAsync(() => api.getPaper(paperId), [paperId]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showConfig, setShowConfig] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [copied, setCopied] = useState(false);
  const [config, setConfig] = useState<RetrievalConfig>({});
  const { isAdvanced } = useViewMode();

  const ask = async (q: string) => {
    if (!q.trim()) return;
    setBusy(true);
    setError(null);
    setAnswer(null);
    try {
      const a = await api.chat(paperId, q.trim(), Object.keys(config).length ? config : undefined);
      setAnswer(a);
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
          { label: "Ask" },
        ]}
      />
      <PageHeader
        kicker="Grounded question answering"
        title="Ask the paper"
        sub={
          isAdvanced
            ? "Advanced — tune retrieval, inspect timings, and copy raw answer JSON."
            : "Answers come from the paper only — every sentence carries a page-level citation."
        }
        right={
          <Link
            to={`/papers/${paperId}/retrieve`}
            className="px-4 py-2 text-sm font-medium border border-line rounded-sm hover:bg-panel hover:border-accent/40 hover:-translate-y-[1px] transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
          >
            Raw retrieval →
          </Link>
        }
      />

      <Card className="mb-6 animate-rise transition-shadow duration-200 hover:shadow-lift">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void ask(question);
          }}
        >
          <label htmlFor="ask-q" className="sr-only">
            Question about this paper
          </label>
          <textarea
            id="ask-q"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={2}
            placeholder="e.g. How many parameters does the generator have?"
            className="w-full resize-none bg-transparent text-base outline-none placeholder:text-ink-faint focus:placeholder:text-ink-mute"
          />
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-line">
            {isAdvanced ? (
              <button
                type="button"
                onClick={() => setShowConfig((s) => !s)}
                aria-expanded={showConfig}
                className="text-xs text-ink-faint hover:text-ink cursor-pointer transition-colors"
              >
                {showConfig ? "hide retrieval settings" : "retrieval settings"}
              </button>
            ) : (
              <span className="text-xs text-ink-faint">Simplified — best settings chosen automatically</span>
            )}
            <Button type="submit" busy={busy} disabled={!question.trim()}>
              {busy ? "Assembling evidence…" : "Ask"}
            </Button>
          </div>
        </form>
        {isAdvanced && showConfig && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-4 border-t border-line animate-fadein">
            <NumField label="dense top-n" value={config.dense_top_n} onChange={(v) => setConfig((c) => ({ ...c, dense_top_n: v }))} />
            <NumField label="sparse top-n" value={config.sparse_top_n} onChange={(v) => setConfig((c) => ({ ...c, sparse_top_n: v }))} />
            <NumField label="RRF k" value={config.rrf_k} onChange={(v) => setConfig((c) => ({ ...c, rrf_k: v }))} />
            <NumField label="final top-k" value={config.final_top_k} onChange={(v) => setConfig((c) => ({ ...c, final_top_k: v }))} />
            <label className="flex items-center gap-2 text-xs text-ink-soft">
              <input
                type="checkbox"
                checked={config.use_dense ?? true}
                onChange={(e) => setConfig((c) => ({ ...c, use_dense: e.target.checked }))}
              />
              dense
            </label>
            <label className="flex items-center gap-2 text-xs text-ink-soft">
              <input
                type="checkbox"
                checked={config.use_sparse ?? true}
                onChange={(e) => setConfig((c) => ({ ...c, use_sparse: e.target.checked }))}
              />
              sparse (BM25)
            </label>
            <label className="flex items-center gap-2 text-xs text-ink-soft">
              <input
                type="checkbox"
                checked={config.use_reranker ?? true}
                onChange={(e) => setConfig((c) => ({ ...c, use_reranker: e.target.checked }))}
              />
              rerank
            </label>
          </div>
        )}
      </Card>

      {!answer && !busy && !error && (
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => {
                setQuestion(s);
                void ask(s);
              }}
              className="px-3 py-1.5 text-xs border border-line rounded-sm text-ink-soft hover:bg-panel hover:border-accent/40 transition-colors cursor-pointer"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {error && (
        <div className="mt-6">
          <ErrorNote message={error} />
        </div>
      )}

      {busy && (
        <Card>
          <Skeleton lines={4} />
          <div className="mt-2 text-xs text-ink-faint">Assembling evidence — dense + BM25 → RRF → rerank…</div>
        </Card>
      )}

      {answer && (
        <div className="mt-2 space-y-4 animate-rise">
          <Card className="transition-shadow duration-200 hover:shadow-lift">
            <div className="flex flex-wrap items-center gap-2 mb-4">
              <Badge tone="accent">{answer.mode === "llm" ? "LLM answer" : "extractive (offline)"}</Badge>
              <Badge tone={sourceTone(answer.confidence)}>{answer.confidence.replace(/_/g, " ")}</Badge>
              <Badge
                tone={
                  answer.sufficiency === "sufficient"
                    ? "ok"
                    : answer.sufficiency === "partial"
                    ? "warn"
                    : "danger"
                }
              >
                {answer.sufficiency}
              </Badge>
              {isAdvanced && answer.stages && (
                <span className="ml-auto text-[11px] text-ink-faint font-mono">
                  {fmtMs(answer.stages.total_ms)} total
                  {answer.tokens_used != null ? ` · ${answer.tokens_used} tok` : ""}
                </span>
              )}
              {isAdvanced && (
                <span className="flex gap-2 ml-auto">
                  <button
                    type="button"
                    onClick={() => setShowRaw((s) => !s)}
                    aria-expanded={showRaw}
                    className="text-[11px] text-ink-faint hover:text-ink underline underline-offset-2 cursor-pointer"
                  >
                    {showRaw ? "hide raw JSON" : "raw JSON"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      void navigator.clipboard?.writeText(JSON.stringify(answer, null, 2)).then(
                        () => {
                          setCopied(true);
                          setTimeout(() => setCopied(false), 1400);
                        },
                        () => {}
                      );
                    }}
                    className="text-[11px] border border-line rounded-sm px-2 py-0.5 text-ink-soft hover:bg-panel transition-colors cursor-pointer"
                  >
                    {copied ? "Copied" : "Copy"}
                  </button>
                </span>
              )}
            </div>
            <p className="text-[15px] leading-relaxed whitespace-pre-wrap max-w-3xl">{answer.answer}</p>
            {isAdvanced && showRaw && (
              <div className="mt-4 animate-fadein">
                <Code>{JSON.stringify(answer, null, 2)}</Code>
              </div>
            )}
          </Card>

          {answer.citations.length > 0 && (
            <Card>
              <h3 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mb-3">
                Citations ({answer.citations.length})
              </h3>
              <ol className="space-y-3">
                {answer.citations.map((c, i) => (
                  <li key={c.chunk_id + i} className="flex gap-3 group">
                    <span className="font-mono text-xs text-ink-faint pt-0.5 w-5 shrink-0">
                      [{i + 1}]
                    </span>
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        <span className="font-medium text-accent">
                          {c.page_range ?? (c.page != null ? `p.${c.page}` : "p.?")}
                        </span>
                        <span className="text-ink-faint truncate">{c.section}</span>
                        {isAdvanced && c.relevance != null && (
                          <span className="font-mono text-ink-faint">
                            rel {c.relevance.toFixed(2)}
                          </span>
                        )}
                        {isAdvanced && (
                          <span className="font-mono text-[10.5px] text-ink-faint truncate opacity-0 group-hover:opacity-100 transition-opacity">
                            {c.chunk_id}
                          </span>
                        )}
                      </div>
                      {c.quote && (
                        <blockquote className="text-[13px] text-ink-soft border-l-2 border-accent/30 pl-3 mt-1.5 italic leading-relaxed">
                          “{c.quote}”
                        </blockquote>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
            </Card>
          )}
        </div>
      )}
    </>
  );
}

function NumField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number | null | undefined;
  onChange: (v: number | null) => void;
}) {
  return (
    <label className="text-xs text-ink-soft">
      <span className="block mb-1">{label}</span>
      <input
        type="number"
        value={value ?? ""}
        placeholder="default"
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        className="w-full border border-line rounded-sm px-2 py-1.5 text-sm bg-card focus:outline-none focus:border-accent"
      />
    </label>
  );
}
