import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { useAsync, useJob, fmtDate, fmtNum } from "../hooks";
import {
  Badge,
  Breadcrumbs,
  Button,
  Card,
  Code,
  Empty,
  ErrorNote,
  Loading,
  PageHeader,
  Table,
  Td,
} from "../components/ui";
import type { RunRecord, ValidationStage } from "../types";

const RUN_KINDS = ["smoke", "tests", "train"] as const;

export default function ImplementationDetail() {
  const { implId = "" } = useParams();
  const { data: impl, loading, error } = useAsync(() => api.getImplementation(implId), [implId]);
  const [selected, setSelected] = useState<string | null>(null);
  const [file, setFile] = useState<{ path: string; content: string } | null>(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [fileErr, setFileErr] = useState<string | null>(null);
  const [runJobId, setRunJobId] = useState<string | null>(null);
  const runJob = useJob(runJobId);
  const { data: runs } = useAsync(
    () =>
      impl
        ? api.listRuns(impl.paper_id).then((rs) => rs.filter((r) => r.implementation_id === impl.id))
        : Promise.resolve([]),
    [impl?.id, runJob?.status]
  );
  const { data: paperJobs } = useAsync(
    () => (impl?.paper_id ? api.listJobs(impl.paper_id) : Promise.resolve([])),
    [impl?.paper_id]
  );
  // validation stages live in the job result that generated this project
  const genJob = useMemo(() => {
    const list = paperJobs ?? [];
    return (
      list.find((j) => {
        const r = j.result as { implementation_id?: string } | null;
        return j.kind === "implement" && r?.implementation_id === impl?.id;
      }) ?? null
    );
  }, [paperJobs, impl?.id]);
  const validationStages = useMemo(() => {
    const r = genJob?.result as { validation?: { stages?: ValidationStage[] } } | null;
    return r?.validation?.stages ?? [];
  }, [genJob]);

  const files = useMemo(() => impl?.files ?? [], [impl]);
  useEffect(() => {
    if (selected === null && files.length > 0) {
      const first = files.find((f) => f.path.endsWith(".py")) ?? files[0];
      setSelected(first.path);
    }
  }, [files, selected]);
  useEffect(() => {
    if (!selected || !impl) return;
    let alive = true;
    setFileLoading(true);
    setFileErr(null);
    api
      .getFile(impl.id, selected)
      .then((f) => alive && setFile({ path: f.path, content: f.content }))
      .catch((e: Error) => alive && setFileErr(e.message))
      .finally(() => alive && setFileLoading(false));
    return () => {
      alive = false;
    };
  }, [selected, impl?.id]);

  if (loading) return <Loading label="Loading implementation…" />;
  if (error) return <ErrorNote message={error} />;
  if (!impl) return <Empty title="Implementation not found" />;

  const run = async (kind: (typeof RUN_KINDS)[number]) => {
    setRunJobId(null);
    try {
      const { job_id } = await api.runImplementation(impl.id, kind);
      setRunJobId(job_id);
    } catch {
      /* error surfaces via job state */
    }
  };

  const runDone = runJob && runJob.status !== "running";
  const runOk = runJob?.status === "done";

  return (
    <>
      <Breadcrumbs
        items={[
          { label: "Papers", to: "/papers" },
          { label: impl.paper_id, to: `/papers/${impl.paper_id}` },
          { label: "Implementations", to: `/papers/${impl.paper_id}/implementations` },
          { label: impl.id },
        ]}
      />
      <PageHeader
        kicker={`${impl.file_count} files · ${fmtDate(impl.created_at)}`}
        title={impl.id}
        sub={
          <>
            From spec{" "}
            {impl.spec_id ? (
              <Link className="text-accent hover:underline" to={`/specs/${impl.spec_id}`}>
                {impl.spec_id}
              </Link>
            ) : (
              "—"
            )}
            . Generated code is untrusted: it runs only in the sandbox (time + memory limits,
            scrubbed env).
          </>
        }
        right={
          <div className="flex items-center gap-2">
            <span className="text-xs text-ink-faint mr-1">sandbox run:</span>
            {RUN_KINDS.map((k) => (
              <Button
                key={k}
                kind={k === "smoke" ? "primary" : "ghost"}
                onClick={() => run(k)}
                busy={runJob?.status === "queued" || runJob?.status === "running"}
              >
                {k}
              </Button>
            ))}
          </div>
        }
      />

      {runJobId && !runDone && (
        <Card className="mb-5 animate-rise">
          <div className="text-sm">
            Running <strong>{runJob?.kind}</strong> in sandbox… {runJob?.detail ?? ""}
          </div>
          <div className="h-1 bg-panel rounded-full mt-3 overflow-hidden border border-line/60">
            <div className="h-full bg-accent transition-all duration-300" style={{ width: `${runJob?.progress ?? 8}%` }} />
          </div>
        </Card>
      )}
      {runJobId && runDone && (
        <Card className={`mb-5 animate-rise ${runOk ? "" : "border-danger/40"}`}>
          <div className="flex items-center justify-between gap-4">
            <span className="text-sm">
              {runOk ? (
                <span className="text-ok">
                  Run finished.{" "}
                  {(() => {
                    const r = (runJob?.result ?? {}) as { run?: RunRecord };
                    if (!r.run) return null;
                    const m = Object.entries(r.run.metrics ?? {});
                    return m.length ? (
                      <>
                        metrics:{" "}
                        {m.map(([k, v]) => (
                          <span key={k} className="font-mono">
                            {k}={String(v)}{" "}
                          </span>
                        ))}
                      </>
                    ) : (
                      "no METRICS line emitted."
                    );
                  })()}
                </span>
              ) : (
                <span className="text-danger">Run failed: {runJob?.error ?? "see log"}</span>
              )}
            </span>
            <Button kind="ghost" onClick={() => setRunJobId(null)}>
              Dismiss
            </Button>
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[16rem_1fr] gap-6 items-start">
        <div className="space-y-5">
          <Card>
            <h3 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mb-3">
              Validation (generation job)
            </h3>
            {validationStages.length === 0 ? (
              <p className="text-sm text-ink-faint">
                No generation job found for this project — validation stages are recorded on the
                job that created it.
              </p>
            ) : (
              <ul className="space-y-3">
                {validationStages.map((s) => (
                  <li key={s.name}>
                    <div className="flex items-center gap-2 text-sm">
                      <span
                        aria-hidden
                        className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                          s.status.startsWith("pass")
                            ? "bg-ok"
                            : s.status.startsWith("fail")
                            ? "bg-danger"
                            : s.status.startsWith("skip")
                            ? "bg-warn"
                            : "bg-ink-faint"
                        }`}
                      />
                      <span className="text-ink-soft">{s.name}</span>
                      <Badge tone={s.status.startsWith("pass") ? "ok" : s.status.startsWith("fail") ? "danger" : "neutral"}>
                        {s.status}
                      </Badge>
                    </div>
                    <div className="text-xs text-ink-faint mt-1 ml-3.5 break-words">{s.detail}</div>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card pad={false}>
            <h3 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium p-4 pb-2">
              Files
            </h3>
            <ul className="pb-2">
              {files.map((f) => (
                <li key={f.path}>
                  <button
                    onClick={() => setSelected(f.path)}
                    className={`w-full text-left px-4 py-1.5 font-mono text-[12px] truncate transition-colors cursor-pointer ${
                      selected === f.path
                        ? "bg-panel text-ink font-medium border-l-2 border-accent"
                        : "text-ink-soft hover:bg-panel/60 border-l-2 border-transparent"
                    }`}
                  >
                    {f.path}
                  </button>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        <div>
          {fileErr ? (
            <ErrorNote message={fileErr} />
          ) : fileLoading ? (
            <Loading label="Loading file…" />
          ) : file ? (
            <Card pad={false} className="animate-fadein">
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
                <code className="font-mono text-xs text-ink-soft">{file.path}</code>
                <span className="text-[11px] text-ink-faint">{file.content.length} chars</span>
              </div>
              <Code>{file.content}</Code>
            </Card>
          ) : (
            <Empty title="Select a file" />
          )}
        </div>
      </div>

      <div className="mt-8">
        <h2 className="font-display text-xl mb-3">Runs</h2>
        {(runs ?? []).length === 0 ? (
          <Empty title="No sandbox runs yet" hint="Use smoke / tests / train above." />
        ) : (
          <Card pad={false}>
            <Table
              head={[
                { label: "Run" },
                { label: "Kind" },
                { label: "Status" },
                { label: "Metrics" },
                { label: "Duration" },
                { label: "Started" },
              ]}
            >
              {(runs ?? []).map((r) => (
                <tr key={r.id} className="hover:bg-panel/50 transition-colors">
                  <Td>
                    <span className="font-mono text-xs">{r.id}</span>
                  </Td>
                  <Td>{r.kind}</Td>
                  <Td>
                    <Badge tone={r.status === "ok" ? "ok" : r.status === "timeout" ? "warn" : "danger"}>
                      {r.status}
                    </Badge>
                  </Td>
                  <Td>
                    <span className="font-mono text-xs">
                      {Object.entries(r.metrics ?? {})
                        .map(([k, v]) => `${k}=${typeof v === "number" ? fmtNum(v) : String(v)}`)
                        .join("  ") || "—"}
                    </span>
                  </Td>
                  <Td>
                    <span className="font-mono text-xs">{(r.duration_ms / 1000).toFixed(1)}s</span>
                  </Td>
                  <Td>{fmtDate(r.started_at)}</Td>
                </tr>
              ))}
            </Table>
          </Card>
        )}
      </div>
    </>
  );
}


