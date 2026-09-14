import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { useAsync, fmtDate } from "../hooks";
import {
  Badge,
  Breadcrumbs,
  Card,
  Empty,
  ErrorNote,
  Loading,
  PageHeader,
  sourceTone,
} from "../components/ui";
import type { ReqStatus, Spec } from "../types";

const STATUSES: ReqStatus[] = ["open", "accepted", "rejected", "implemented"];

export default function SpecDetail() {
  const { specId = "" } = useParams();
  const { data: spec, loading, error, reload } = useAsync(() => api.getSpec(specId), [specId]);
  const [statusOverrides, setStatusOverrides] = useState<Record<string, ReqStatus>>({});
  const [filter, setFilter] = useState<"all" | ReqStatus | "EXPLICIT" | "INFERRED">("all");

  if (loading) return <Loading label="Loading spec…" />;
  if (error) return <ErrorNote message={error} />;
  if (!spec) return <Empty title="Spec not found" />;

  return (
    <SpecView
      spec={spec}
      statusOverrides={statusOverrides}
      setStatusOverrides={setStatusOverrides}
      filter={filter}
      setFilter={setFilter}
      reload={reload}
    />
  );
}

function SpecView({
  spec,
  statusOverrides,
  setStatusOverrides,
  filter,
  setFilter,
  reload,
}: {
  spec: Spec;
  statusOverrides: Record<string, ReqStatus>;
  setStatusOverrides: (fn: (p: Record<string, ReqStatus>) => Record<string, ReqStatus>) => void;
  filter: string;
  setFilter: (f: "all" | ReqStatus | "EXPLICIT" | "INFERRED") => void;
  reload: () => void;
}) {
  const counts = useMemo(() => {
    const c = { EXPLICIT: 0, INFERRED: 0, EXTERNAL: 0, UNKNOWN: 0, total: spec.requirements.length };
    for (const r of spec.requirements) c[r.source.toUpperCase() as keyof typeof c] += 1;
    return c;
  }, [spec.requirements]);

  const statusOf = (reqId: string, fallback: ReqStatus) => statusOverrides[reqId] ?? fallback;

  const shown = spec.requirements.filter((r) => {
    if (filter === "all") return true;
    if (filter === "EXPLICIT" || filter === "INFERRED") return r.source === filter;
    return statusOf(r.req_id, r.status as ReqStatus) === filter;
  });

  const setStatus = async (reqId: string, status: ReqStatus) => {
    setStatusOverrides((p) => ({ ...p, [reqId]: status }));
    try {
      await api.setRequirementStatus(spec.id, reqId, status);
      reload();
    } catch {
      /* optimistic update stands; error surfaces on reload */
    }
  };

  const nAccepted = spec.requirements.filter(
    (r) => statusOf(r.req_id, r.status as ReqStatus) !== "rejected"
  ).length;

  return (
    <>
      <Breadcrumbs
        items={[
          { label: "Papers", to: "/papers" },
          { label: spec.paper_id, to: `/papers/${spec.paper_id}` },
          { label: "Specs", to: `/papers/${spec.paper_id}/specs` },
          { label: spec.id },
        ]}
      />
      <PageHeader
        kicker={`${spec.generated_by ?? "unknown generator"} · v${spec.version} · ${fmtDate(spec.created_at)}`}
        title={spec.title || "Implementation specification"}
        sub={spec.summary}
        right={
          <Link
            to={`/papers/${spec.paper_id}/implementations`}
            className="px-4 py-2 text-sm font-medium bg-ink text-paper rounded-sm hover:bg-ink-soft transition-colors"
          >
            Generate code →
          </Link>
        }
      />

      {spec.open_questions.length > 0 && (
        <Card className="mb-6 border-warn/30 bg-warn-soft/40 animate-rise">
          <h3 className="text-[11px] uppercase tracking-[0.16em] text-warn font-medium mb-2">
            Reported gaps — the paper does not specify
          </h3>
          <ul className="text-sm text-ink-soft list-disc pl-5 space-y-1">
            {spec.open_questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </Card>
      )}

      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mr-1">
            {counts.total} requirements · {nAccepted} feeding codegen
          </span>
          {(["all", "open", "accepted", "rejected", "implemented", "EXPLICIT", "INFERRED"] as const).map(
            (f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-2.5 py-1 text-[11px] rounded-sm border transition-colors cursor-pointer ${
                  filter === f
                    ? "border-accent bg-accent-soft text-accent font-medium"
                    : "border-line text-ink-faint hover:text-ink"
                }`}
              >
                {f}
              </button>
            )
          )}
        </div>
        <span className="text-xs text-ink-faint">
          provenance:{" "}
          {Object.entries(counts)
            .filter(([k]) => k !== "total")
            .map(([k, v]) => `${k} ${v}`)
            .join(" · ")}
        </span>
      </div>

      {shown.length === 0 ? (
        <Empty title="No requirements match this filter" />
      ) : (
        <ul className="space-y-3">
          {shown.map((r) => (
            <Card key={r.id} className="animate-rise">
              <div className="flex items-start justify-between gap-6">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2 mb-2">
                    <span className="font-mono text-xs text-ink-faint">{r.req_id}</span>
                    <Badge tone="neutral">{r.type}</Badge>
                    <Badge tone={sourceTone(r.source)}>{r.source}</Badge>
                    {r.source_page != null && <Badge tone="neutral">p.{r.source_page}</Badge>}
                    {r.confidence != null && (
                      <span className="text-[11px] text-ink-faint font-mono">
                        conf {r.confidence.toFixed(2)}
                      </span>
                    )}
                  </div>
                  <p className="text-[15px] leading-snug font-medium">{r.requirement}</p>
                  {r.implication && (
                    <p className="text-[13px] text-ink-soft mt-1.5">{r.implication}</p>
                  )}
                  {r.source_quote && (
                    <blockquote className="mt-3 border-l-2 border-accent/40 pl-3 text-[13px] italic text-ink-soft leading-relaxed">
                      “{r.source_quote}”
                      {r.source_section && (
                        <span className="not-italic text-ink-faint"> — {r.source_section}</span>
                      )}
                    </blockquote>
                  )}
                </div>
                <div className="shrink-0 flex flex-col items-end gap-2">
                  <div className="flex rounded-sm border border-line overflow-hidden">
                    {STATUSES.map((s) => {
                      const cur = statusOf(r.req_id, r.status as ReqStatus);
                      return (
                        <button
                          key={s}
                          onClick={() => setStatus(r.req_id, s)}
                          aria-pressed={cur === s}
                          className={`px-2.5 py-1.5 text-[11px] transition-colors cursor-pointer ${
                            cur === s
                              ? s === "rejected"
                                ? "bg-danger-soft text-danger font-medium"
                                : "bg-panel text-ink font-medium"
                              : "text-ink-faint hover:text-ink hover:bg-panel/60"
                          }`}
                        >
                          {s}
                        </button>
                      );
                    })}
                  </div>
                  {r.notes && <span className="text-[11px] text-ink-faint max-w-[12rem] text-right">{r.notes}</span>}
                </div>
              </div>
            </Card>
          ))}
        </ul>
      )}
    </>
  );
}
