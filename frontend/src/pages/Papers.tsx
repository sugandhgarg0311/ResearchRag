import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useAsync, useJob, fmtDate } from "../hooks";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Loading,
  PageHeader,
  Table,
  Td,
} from "../components/ui";

export default function Papers() {
  const { data: papers, loading, error, reload } = useAsync(() => api.listPapers(), []);
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const job = useJob(jobId);
  const [removed, setRemoved] = useState<string | null>(null);

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setBusy(true);
    setUploadErr(null);
    try {
      const { job_id } = await api.uploadPaper(f);
      setJobId(job_id);
      setRemoved(null);
    } catch (e) {
      setUploadErr((e as Error).message);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const onDelete = async (id: string) => {
    if (!confirm("Delete this paper and all its indexes?")) return;
    await api.deletePaper(id);
    setRemoved(id);
    reload();
  };

  const done = job && job.status !== "running";
  return (
    <>
      <PageHeader
        kicker="Library"
        title="Papers"
        sub="Each PDF is parsed into pages, sections, and provenance blocks, then chunked and indexed."
        right={
          <>
            <input
              ref={fileRef}
              type="file"
              accept="application/pdf"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
            <Button kind="ghost" onClick={reload} disabled={busy}>
              Refresh
            </Button>
            <Button busy={busy} onClick={() => fileRef.current?.click()}>
              Upload PDF
            </Button>
          </>
        }
      />

      {uploadErr && (
        <div className="mb-4">
          <ErrorNote message={uploadErr} />
        </div>
      )}
      {jobId && (
        <div className="mb-4">
          {done ? (
            <Card>
              {job?.status === "done" ? (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-sm text-ok">
                    Ingestion complete —{" "}
                    {(() => {
                      const r = (job?.result ?? {}) as { paper_id?: string };
                      return r.paper_id ? (
                        <Link className="underline" to={`/papers/${r.paper_id}`}>
                          open paper
                        </Link>
                      ) : (
                        <span>view library</span>
                      );
                    })()}
                  </span>
                  <Button kind="ghost" onClick={() => setJobId(null)}>
                    Dismiss
                  </Button>
                </div>
              ) : (
                <div className="flex items-center justify-between gap-4">
                  <span className="text-sm text-danger">
                    Ingestion failed: {job?.error ?? "unknown error"}
                  </span>
                  <Button kind="ghost" onClick={() => setJobId(null)}>
                    Dismiss
                  </Button>
                </div>
              )}
            </Card>
          ) : (
            <Card>
              <div className="flex items-center justify-between">
                <span className="text-sm">
                  Ingesting… {job?.detail ?? "parsing"} ({Math.round(job?.progress ?? 0)}%)
                </span>
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

      {removed && (
        <div className="mb-4">
          <Card>
            <div className="flex items-center justify-between">
              <span className="text-sm text-ink-soft">
                Paper <code className="font-mono text-xs">{removed}</code> deleted.
              </span>
              <Button kind="ghost" onClick={() => setRemoved(null)}>
                Dismiss
              </Button>
            </div>
          </Card>
        </div>
      )}

      {error && <ErrorNote message={error} />}
      {loading ? (
        <Loading label="Loading papers…" />
      ) : (papers ?? []).length === 0 ? (
        <Empty title="Library is empty" hint="Upload a research PDF to get started." />
      ) : (
        <Card pad={false}>
          <Table
            head={[
              { label: "Title" },
              { label: "Pages" },
              { label: "Status" },
              { label: "Parser" },
              { label: "Ingested" },
              { label: "", className: "text-right" },
            ]}
          >
            {(papers ?? []).map((p) => (
              <tr key={p.id} className="hover:bg-panel/50 transition-colors">
                <Td className="min-w-[16rem]">
                  <Link to={`/papers/${p.id}`} className="font-medium hover:text-accent">
                    {p.title || p.filename}
                  </Link>
                  <div className="text-xs text-ink-faint font-mono mt-0.5">{p.id}</div>
                </Td>
                <Td>{p.page_count > 0 ? p.page_count : "—"}</Td>
                <Td>
                  <Badge
                    tone={
                      p.status === "ready" ? "ok" : p.status === "failed" ? "danger" : "warn"
                    }
                  >
                    {p.status}
                  </Badge>
                </Td>
                <Td>{p.parser ?? "—"}</Td>
                <Td>{fmtDate(p.created_at)}</Td>
                <Td className="text-right">
                  <button
                    onClick={() => onDelete(p.id)}
                    className="text-xs text-ink-faint hover:text-danger cursor-pointer transition-colors"
                  >
                    delete
                  </button>
                </Td>
              </tr>
            ))}
          </Table>
        </Card>
      )}
    </>
  );
}
