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
  PageHeader,
  Table,
  Td,
} from "../components/ui";
import type { Experiment } from "../types";

interface Row {
  strategy: string;
  retrieval: string;
  rerank: string;
  top_k: number;
  [k: string]: unknown;
}

const METRIC_COLS = ["recall@k", "precision@k", "mrr", "ndcg@k"] as const;

export default function Experiments() {
  const { paperId = "" } = useParams();
  const paper = useAsync(() => api.getPaper(paperId), [paperId]);
  const { data: experiments, loading, error, reload } = useAsync(
    () => api.listExperiments(paperId),
    [paperId]
  );
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId);

  const start = async () => {
    try {
      const { job_id } = await api.runExperiments(paperId);
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
          { label: "Experiments" },
        ]}
      />
      <PageHeader
        kicker="Retrieval experiments"
        title="Architecture comparison"
        sub="Sweeps chunking strategy × retrieval mode (dense / sparse / hybrid) × RRF × rerank × top-K on the fixed eval dataset, in-memory. Shows empirically why the chosen configuration is used."
        right={
          <Button onClick={start} busy={job?.status === "queued" || job?.status === "running"}>
            Run experiment
          </Button>
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
                      {(job?.result as { rows?: unknown[] })?.rows
                        ? `Done — ${(job?.result as { rows?: unknown[] }).rows?.length} configurations compared.`
                        : "Experiment complete."}
                    </span>
                  ) : (
                    <span className="text-danger">Experiment failed: {job?.error}</span>
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
                Sweeping configurations… {job?.detail ?? ""} ({Math.round(job?.progress ?? 0)}%)
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
        <Loading label="Loading experiments…" />
      ) : (experiments ?? []).length === 0 ? (
        <Empty
          title="No experiments yet"
          hint="Run the sweep to compare dense vs sparse vs hybrid retrieval across chunking strategies."
        />
      ) : (
        <div className="space-y-5">
          {(experiments ?? []).map((e) => (
            <ExperimentCard key={e.id} exp={e} />
          ))}
        </div>
      )}
    </>
  );
}

function ExperimentCard({ exp }: { exp: Experiment }) {
  const rows = ((exp.results as { rows?: Row[] })?.rows ?? []) as Row[];
  const cfg = exp.config as { n_questions?: number; k_swept?: number[] };
  const [filter, setFilter] = useState<string>("all");

  const strategies = [...new Set(rows.map((r) => r.strategy))];
  const shown = rows
    .map((r, i) => ({ r, i }))
    .filter(({ r }) => filter === "all" || r.strategy === filter);

  // best row index per (retrieval, top_k, rerank) by recall — highlights winners
  const bestIdx = new Map<string, number>();
  rows.forEach((r, i) => {
    const key = `${r.retrieval}|${r.top_k}|${r.rerank}`;
    const cur = bestIdx.get(key);
    if (cur === undefined || (r["recall@k"] ?? 0) > (rows[cur]["recall@k"] ?? 0)) {
      bestIdx.set(key, i);
    }
  });

  return (
    <Card className="animate-rise">
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <span className="font-mono text-xs text-ink-faint">{exp.id}</span>
        <Badge tone="neutral">{exp.name}</Badge>
        <Badge tone="accent">{cfg.n_questions ?? "?"} questions</Badge>
        <span className="text-xs text-ink-faint ml-auto">{fmtDate(exp.created_at)}</span>
      </div>
      <div className="flex gap-1.5 mb-4">
        {["all", ...strategies].map((s) => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={`px-2.5 py-1 text-[11px] rounded-sm border transition-colors cursor-pointer ${
              filter === s
                ? "border-accent bg-accent-soft text-accent font-medium"
                : "border-line text-ink-faint hover:text-ink"
            }`}
          >
            {s}
          </button>
        ))}
      </div>
      <Table
        head={[
          { label: "Strategy" },
          { label: "Retrieval" },
          { label: "Rerank" },
          { label: "K" },
          ...METRIC_COLS.map((c) => ({ label: c })),
        ]}
      >
        {shown.map(({ r, i }) => {
          const key = `${r.retrieval}|${r.top_k}|${r.rerank}`;
          const isBest = bestIdx.get(key) === i;
          return (
            <tr key={i} className={isBest ? "bg-accent-soft/40" : "hover:bg-panel/50 transition-colors"}>
              <Td>
                <span className="font-medium">{r.strategy}</span>
                {isBest && <Badge tone="accent">top</Badge>}
              </Td>
              <Td>{r.retrieval}</Td>
              <Td>{r.rerank}</Td>
              <Td>
                <span className="font-mono text-xs">{r.top_k}</span>
              </Td>
              {METRIC_COLS.map((c) => (
                <Td key={c}>
                  <span className="font-mono text-xs tabular-nums">
                    {num(r[c])}
                  </span>
                </Td>
              ))}
            </tr>
          );
        })}
      </Table>
      <p className="text-[11px] text-ink-faint mt-3">
        Highlighted rows are the best recall for each (retrieval, K, rerank) combination.
      </p>
    </Card>
  );
}

function num(v: unknown): string {
  return typeof v === "number" ? v.toFixed(4) : "—";
}
