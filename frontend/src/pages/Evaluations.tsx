import { useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import { useAsync, useJob, fmtDate } from "../hooks";
import {
  Badge,
  Breadcrumbs,
  Button,
  Card,
  Empty,
  ErrorNote,
  Loading,
  MetricBar,
  PageHeader,
  Table,
  Td,
} from "../components/ui";
import type { EvalRun } from "../types";

const RETRIEVAL_METRICS = [
  "recall@k",
  "precision@k",
  "mrr",
  "ndcg@k",
  "context_precision@k",
  "context_recall@k",
] as const;
const ANSWER_METRICS = ["citation_correctness", "groundedness", "answer_relevance"] as const;

export default function Evaluations() {
  const { paperId = "" } = useParams();
  const paper = useAsync(() => api.getPaper(paperId), [paperId]);
  const { data: datasets } = useAsync(() => api.listDatasets(paperId), [paperId]);
  const { data: runs, loading, error, reload } = useAsync(
    () => api.listEvaluations(paperId),
    [paperId]
  );
  const [dataset, setDataset] = useState<string>("");
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId);

  const start = async () => {
    try {
      const { job_id } = await api.runEvaluation(paperId, dataset || undefined);
      setJobId(job_id);
    } catch {
      /* surfaced via job */
    }
  };

  const done = job && job.status !== "running" && job.status !== "queued";

  return (
    <>
      <Breadcrumbs
        items={[
          { label: "Papers", to: "/papers" },
          { label: paper.data?.title || paperId, to: `/papers/${paperId}` },
          { label: "Evaluation" },
        ]}
      />
      <PageHeader
        kicker="Evaluation"
        title="Retrieval & answer metrics"
        sub="Recall@K, Precision@K, MRR, NDCG@K, context precision/recall — plus answer-level citation correctness, groundedness, and relevance. Answer metrics run the offline extractive answerer (deterministic)."
        right={
          <div className="flex items-center gap-2">
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="border border-line rounded-sm px-2.5 py-2 text-sm bg-card max-w-[16rem] cursor-pointer"
            >
              {(datasets ?? []).length === 0 && <option value="">no datasets</option>}
              {(datasets ?? []).map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
            <Button onClick={start} busy={job?.status === "queued" || job?.status === "running"}>
              Run evaluation
            </Button>
          </div>
        }
      />

      {jobId && (
        <div className="mb-4">
          {done ? (
            <Card>
              <div className="flex items-center justify-between">
                <span className="text-sm">
                  {job?.status === "done" ? (
                    <span className="text-ok">
                      {(job?.result as { summary?: { n_questions?: number } })?.summary
                        ? `Done — ${(job?.result as { summary?: { n_questions?: number } }).summary?.n_questions} questions scored.`
                        : "Evaluation complete."}
                    </span>
                  ) : (
                    <span className="text-danger">Evaluation failed: {job?.error}</span>
                  )}
                </span>
                <Button
                  kind="ghost"
                  onClick={() => {
                    setJobId(null);
                    reload();
                  }}
                >
                  Dismiss
                </Button>
              </div>
            </Card>
          ) : (
            <Card>
              <div className="text-sm">
                Scoring questions… {job?.detail ?? ""} ({Math.round(job?.progress ?? 0)}%)
              </div>
              <div className="h-1 bg-panel rounded-full mt-3 overflow-hidden border border-line/60">
                <div
                  className="h-full bg-accent transition-all duration-300"
                  style={{ width: `${job?.progress ?? 0}%` }}
                />
              </div>
            </Card>
          )}
        </div>
      )}

      {error && <ErrorNote message={error} />}
      {loading ? (
        <Loading label="Loading evaluations…" />
      ) : (runs ?? []).length === 0 ? (
        <Empty
          title="No evaluation runs yet"
          hint="Run the bundled eval dataset (32 QA pairs over the sample paper) — gold chunks are resolved against the ingested chunk ids."
        />
      ) : (
        <div className="space-y-5">
          {(runs ?? []).map((r) => (
            <EvalCard key={r.id} run={r} />
          ))}
        </div>
      )}
    </>
  );
}

function EvalCard({ run }: { run: EvalRun }) {
  const m = run.metrics as {
    k?: number;
    n_questions?: number;
    retrieval?: Record<string, number>;
    answers?: Record<string, number>;
  };
  const [openDetail, setOpenDetail] = useState(false);
  const pq = run.per_question as {
    retrieval?: Array<Record<string, unknown>>;
    answers?: Array<Record<string, unknown>>;
  };

  return (
    <Card className="animate-rise">
      <div className="flex flex-wrap items-center gap-2 mb-5">
        <span className="font-mono text-xs text-ink-faint">{run.id}</span>
        <Badge tone="neutral">{run.dataset}</Badge>
        <Badge tone="accent">k={m.k ?? (typeof run.config.k === "number" ? run.config.k : 10)}</Badge>
        <Badge tone="neutral">{m.n_questions ?? "?"} questions</Badge>
        <span className="text-xs text-ink-faint ml-auto">
          {String(run.config.chunk_strategy ?? "—")} · {fmtDate(run.created_at)}
        </span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-8 gap-y-4">
        <div>
          <h4 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mb-3">
            Retrieval
          </h4>
          <div className="space-y-3">
            {RETRIEVAL_METRICS.map((mk) => (
              <MetricBar key={mk} label={mk} value={m.retrieval?.[mk]} />
            ))}
          </div>
        </div>
        <div>
          <h4 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mb-3">
            Answers
          </h4>
          <div className="space-y-3">
            {ANSWER_METRICS.map((mk) => (
              <MetricBar key={mk} label={mk} value={m.answers?.[mk]} />
            ))}
          </div>
        </div>
        <div>
          <button
            onClick={() => setOpenDetail((v) => !v)}
            className="text-xs text-accent hover:underline cursor-pointer"
          >
            {openDetail ? "hide per-question detail" : "show per-question detail"}
          </button>
        </div>
      </div>
      {openDetail && pq?.retrieval && (
        <div className="mt-5 pt-4 border-t border-line animate-fadein">
          <h4 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mb-2">
            Per-question retrieval
          </h4>
          <Table
            head={[
              { label: "Question" },
              { label: "recall@k" },
              { label: "mrr" },
              { label: "ndcg@k" },
            ]}
          >
            {pq.retrieval.map((q, i) => (
              <tr key={i}>
                <Td className="max-w-[28rem]">
                  <span className="line-clamp-1">{String(q.question ?? "")}</span>
                </Td>
                <Td>
                  <span className="font-mono text-xs">{num(q["recall@k"])}</span>
                </Td>
                <Td>
                  <span className="font-mono text-xs">{num(q.mrr)}</span>
                </Td>
                <Td>
                  <span className="font-mono text-xs">{num(q["ndcg@k"])}</span>
                </Td>
              </tr>
            ))}
          </Table>
        </div>
      )}
    </Card>
  );
}

function num(v: unknown): string {
  return typeof v === "number" ? v.toFixed(4) : "—";
}
