import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { useAsync } from "../hooks";
import { useViewMode } from "../view";
import {
  Badge,
  Button,
  Card,
  Code,
  Empty,
  ErrorNote,
  PageHeader,
  Skeleton,
  sourceTone,
} from "../components/ui";
import type { Answer } from "../types";

const SUGGESTIONS = [
  "What is the main contribution of the paper?",
  "What model architecture does the paper use?",
  "What are the key equations and what do they mean?",
  "What hardware was used for training?",
];

export default function Chat() {
  const [params, setParams] = useSearchParams();
  const { data: papers } = useAsync(() => api.listPapers(), []);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showRaw, setShowRaw] = useState(false);
  const { isAdvanced } = useViewMode();

  const ready = useMemo(() => (papers ?? []).filter((p) => p.status === "ready"), [papers]);
  const paperId = params.get("paper") ?? "";
  const activeId = ready.some((p) => p.id === paperId) ? paperId : ready[0]?.id ?? "";

  const ask = async (q: string) => {
    if (!q.trim() || !activeId) return;
    setBusy(true);
    setError(null);
    setAnswer(null);
    try {
      setAnswer(await api.chat(activeId, q.trim()));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        kicker="Chat"
        title="Chat with your papers"
        sub={
          isAdvanced
            ? "Global chat — pick a paper, ask grounded questions, inspect raw answer JSON."
            : "Pick a paper and ask questions — every answer is grounded in the paper with citations."
        }
        right={<ThemeNote />}
      />

      {ready.length === 0 ? (
        <Empty
          title="No papers ready yet"
          hint={
            <>
              Upload a PDF in the <Link to="/papers" className="text-accent underline">library</Link> first,
              then come back to chat.
            </>
          }
        />
      ) : (
        <>
          <Card className="mb-6 animate-rise transition-shadow duration-200 hover:shadow-lift">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void ask(question);
              }}
            >
              <div className="flex flex-wrap items-center gap-3 mb-3">
                <label htmlFor="chat-paper" className="text-xs text-ink-faint font-medium uppercase tracking-wide">
                  Paper
                </label>
                <select
                  id="chat-paper"
                  value={activeId}
                  onChange={(e) => setParams(e.target.value ? { paper: e.target.value } : {})}
                  className="border border-line rounded-sm px-2.5 py-2 text-sm bg-card max-w-full sm:max-w-[28rem] cursor-pointer focus:outline-none focus:border-accent"
                >
                  {ready.map((p) => (
                    <option key={p.id} value={p.id}>
                      {(p.title || p.filename) + ` (${p.page_count}p)`}
                    </option>
                  ))}
                </select>
                {activeId && (
                  <Link to={`/papers/${activeId}/ask`} className="text-xs text-accent hover:underline">
                    Open paper workspace →
                  </Link>
                )}
              </div>
              <label htmlFor="chat-q" className="sr-only">
                Your question
              </label>
              <textarea
                id="chat-q"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={2}
                placeholder="e.g. What is the main equation of this paper?"
                className="w-full resize-none bg-transparent text-base outline-none placeholder:text-ink-faint"
              />
              <div className="flex items-center justify-end mt-3 pt-3 border-t border-line">
                <Button type="submit" busy={busy} disabled={!question.trim() || !activeId}>
                  {busy ? "Thinking…" : "Ask"}
                </Button>
              </div>
            </form>
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
                  className="px-3 py-1.5 text-xs border border-line rounded-sm text-ink-soft hover:bg-panel hover:border-accent/40 hover:-translate-y-[1px] transition-all cursor-pointer"
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          {busy && (
            <Card>
              <Skeleton lines={4} />
              <div className="mt-2 text-xs text-ink-faint">Reading the paper and assembling evidence…</div>
            </Card>
          )}
          {error && (
            <div className="mt-6">
              <ErrorNote message={error} />
            </div>
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
                  {isAdvanced && (
                    <button
                      type="button"
                      onClick={() => setShowRaw((s) => !s)}
                      aria-expanded={showRaw}
                      className="ml-auto text-[11px] text-ink-faint hover:text-ink underline underline-offset-2 cursor-pointer"
                    >
                      {showRaw ? "hide raw JSON" : "raw JSON"}
                    </button>
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
                      <li key={c.chunk_id + i} className="flex gap-3">
                        <span className="font-mono text-xs text-ink-faint pt-0.5 w-5 shrink-0">[{i + 1}]</span>
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2 text-xs">
                            <span className="font-medium text-accent">
                              {c.page_range ?? (c.page != null ? `p.${c.page}` : "p.?")}
                            </span>
                            <span className="text-ink-faint truncate">{c.section}</span>
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
      )}
    </>
  );
}

function ThemeNote() {
  return (
    <span className="text-[11px] text-ink-faint hidden sm:inline">
      Grounded answers only — never parametric knowledge.
    </span>
  );
}
