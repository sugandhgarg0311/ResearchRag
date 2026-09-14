import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useAsync, fmtDate } from "../hooks";
import { Badge, Card, Empty, ErrorNote, Loading, PageHeader, Table, Td } from "../components/ui";

export default function Jobs() {
  const [paperId, setPaperId] = useState<string>("");
  const { data: papers } = useAsync(() => api.listPapers(), []);
  const { data: jobs, loading, error, reload } = useAsync(
    () => (paperId ? api.listJobs(paperId) : api.listJobs()),
    [paperId]
  );

  // refresh while any visible job is in flight
  const anyActive = (jobs ?? []).some((j) => j.status === "queued" || j.status === "running");
  useEffect(() => {
    if (!anyActive) return;
    const t = setInterval(reload, 1500);
    return () => clearInterval(t);
  }, [anyActive, reload]);

  return (
    <>
      <PageHeader
        kicker="Background work"
        title="Jobs"
        sub="Ingestion, spec generation, code generation, sandbox runs, evaluations, and experiments all run as jobs with progress and results."
        right={
          <select
            value={paperId}
            onChange={(e) => setPaperId(e.target.value)}
            className="border border-line rounded-sm px-2.5 py-2 text-sm bg-card max-w-[18rem] cursor-pointer"
          >
            <option value="">all papers</option>
            {(papers ?? []).map((p) => (
              <option key={p.id} value={p.id}>
                {p.title || p.filename}
              </option>
            ))}
          </select>
        }
      />
      {!loading && !error && (jobs ?? []).length === 0 ? (
        <Empty title={paperId ? "No jobs for this paper yet" : "No jobs yet"} />
      ) : (
        <>
          {error && <ErrorNote message={error} />}
          {loading ? (
            <Loading label="Loading jobs…" />
          ) : (
            <Card pad={false}>
              <Table
                head={[
                  { label: "Job" },
                  { label: "Paper" },
                  { label: "Kind" },
                  { label: "Status" },
                  { label: "Progress" },
                  { label: "Detail / error" },
                  { label: "Started" },
                ]}
              >
                {(jobs ?? []).map((j) => (
                  <tr key={j.id} className="hover:bg-panel/50 transition-colors">
                    <Td>
                      <span className="font-mono text-xs">{j.id}</span>
                    </Td>
                    <Td>
                      {j.paper_id ? (
                        <Link to={`/papers/${j.paper_id}`} className="font-mono text-xs hover:text-accent">
                          {j.paper_id}
                        </Link>
                      ) : (
                        <span className="text-xs text-ink-faint">—</span>
                      )}
                    </Td>
                    <Td>
                      <Badge tone={j.kind === "run" ? "accent" : "neutral"}>{j.kind}</Badge>
                    </Td>
                    <Td>
                      <Badge
                        tone={
                          j.status === "done"
                            ? "ok"
                            : j.status === "failed"
                            ? "danger"
                            : "warn"
                        }
                      >
                        {j.status}
                      </Badge>
                    </Td>
                    <Td>
                      <div className="w-24">
                        <div className="h-1 bg-panel rounded-full overflow-hidden border border-line/60">
                          <div
                            className="h-full bg-accent"
                            style={{ width: `${Math.round(j.progress ?? 0)}%` }}
                          />
                        </div>
                        <span className="text-[10px] font-mono text-ink-faint">
                          {Math.round(j.progress ?? 0)}%
                        </span>
                      </div>
                    </Td>
                    <Td>
                      <span className="text-xs text-ink-soft block max-w-[24rem] truncate">
                        {j.status === "failed" ? j.error : j.detail}
                      </span>
                    </Td>
                    <Td>
                      <span className="text-xs text-ink-faint whitespace-nowrap">
                        {fmtDate(j.created_at)}
                      </span>
                    </Td>
                  </tr>
                ))}
              </Table>
            </Card>
          )}
        </>
      )}
      <p className="text-[11px] text-ink-faint mt-4">
        Job results are stored alongside each job —{" "}
        <Link to="/health" className="underline hover:text-ink">system health</Link> shows the
        pipeline components.
      </p>
    </>
  );
}
