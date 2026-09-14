import { useState } from "react";
import { Link, useParams } from "react-router-dom";
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

export default function Implementations() {
  const { paperId = "" } = useParams();
  const paper = useAsync(() => api.getPaper(paperId), [paperId]);
  const { data: impls, loading, error, reload } = useAsync(
    () => api.listImplementations(paperId),
    [paperId]
  );
  const { data: specs } = useAsync(() => api.listSpecs(paperId), [paperId]);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId);
  const [specId, setSpecId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const generate = async () => {
    setErr(null);
    try {
      const { job_id } = await api.generateImplementation(paperId, specId ?? undefined);
      setJobId(job_id);
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  const done = job && job.status !== "running";
  return (
    <>
      <Breadcrumbs
        items={[
          { label: "Papers", to: "/papers" },
          { label: paper.data?.title || paperId, to: `/papers/${paperId}` },
          { label: "Implementations" },
        ]}
      />
      <PageHeader
        kicker="Paper → code"
        title="Implementations"
        sub="Generated from the implementation spec — never directly from the paper. Validation runs syntax, import, and structure checks; execution happens in the sandbox only."
        right={
          (specs ?? []).length > 0 ? (
            <div className="flex items-center gap-2">
              <select
                value={specId ?? ""}
                onChange={(e) => setSpecId(e.target.value || null)}
                className="border border-line rounded-sm px-2.5 py-2 text-sm bg-card max-w-[16rem] cursor-pointer"
              >
                {(specs ?? []).map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.title || s.id}
                  </option>
                ))}
              </select>
              <Button onClick={generate} busy={job?.status === "running"}>
                Generate project
              </Button>
            </div>
          ) : (
            <Link
              to={`/papers/${paperId}/specs`}
              className="px-4 py-2 text-sm font-medium border border-line rounded-sm hover:bg-panel transition-colors"
            >
              Create a spec first →
            </Link>
          )
        }
      />

      {err && <div className="mb-4"><ErrorNote message={err} /></div>}
      {jobId && (
        <div className="mb-4">
          {done ? (
            <Card>
              <div className="flex items-center justify-between">
                <span className="text-sm">
                  {job?.status === "done" ? (
                    <span className="text-ok">
                      Project ready —{" "}
                      {(() => {
                        const r = (job?.result ?? {}) as { implementation_id?: string; validation?: { ok: boolean } };
                        return r.implementation_id ? (
                          <Link className="underline" to={`/implementations/${r.implementation_id}`}>
                            open project
                          </Link>
                        ) : (
                          "open it"
                        );
                      })()}
                    </span>
                  ) : (
                    <span className="text-danger">Generation failed: {job?.error}</span>
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
                Generating & validating… {job?.detail ?? ""} ({Math.round(job?.progress ?? 0)}%)
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
        <Loading label="Loading implementations…" />
      ) : (impls ?? []).length === 0 ? (
        <Empty
          title="No implementations yet"
          hint="Generate a spec first, then a project from it. The scaffold is always runnable; an LLM (if configured) can enhance individual files."
        />
      ) : (
        <Card pad={false} className="animate-rise">
          <Table
            head={[
              { label: "Project" },
              { label: "Spec" },
              { label: "Files" },
              { label: "Status" },
              { label: "Created" },
              { label: "", className: "text-right" },
            ]}
          >
            {(impls ?? []).map((im) => (
              <tr key={im.id} className="hover:bg-panel/50 transition-colors">
                <Td>
                  <Link to={`/implementations/${im.id}`} className="font-medium hover:text-accent">
                    <span className="font-mono text-sm">{im.id}</span>
                  </Link>
                </Td>
                <Td>
                  {im.spec_id ? (
                    <Link to={`/specs/${im.spec_id}`} className="font-mono text-xs text-accent hover:underline">
                      {im.spec_id}
                    </Link>
                  ) : (
                    "—"
                  )}
                </Td>
                <Td>{im.file_count}</Td>
                <Td>
                  <Badge tone={im.status === "validated" ? "ok" : "neutral"}>{im.status}</Badge>
                </Td>
                <Td>{fmtDate(im.created_at)}</Td>
                <Td className="text-right">
                  <Link to={`/implementations/${im.id}`} className="text-xs text-accent hover:underline">
                    open →
                  </Link>
                </Td>
              </tr>
            ))}
          </Table>
        </Card>
      )}
    </>
  );
}
