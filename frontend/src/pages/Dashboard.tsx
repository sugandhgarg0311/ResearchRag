import { Link } from "react-router-dom";
import { api } from "../api";
import { useAsync, fmtDate } from "../hooks";
import { useViewMode } from "../view";
import { Card, PageHeader, Stat, Badge, Empty, ErrorNote, Skeleton, Stagger, StatusDot, embedderStatus } from "../components/ui";

export default function Dashboard() {
  const { data: health, error: hErr } = useAsync(() => api.health(), []);
  const { data: papers, loading, error } = useAsync(() => api.listPapers(), []);
  const { isAdvanced } = useViewMode();

  if (hErr)
    return <ErrorNote message={`Cannot reach the API: ${hErr}. Start it with “researchrag serve”.`} />;
  if (loading || !health) {
    return (
      <>
        <PageHeader kicker="Overview" title="Dashboard" sub="Loading your evidence-first workspace…" />
        <Card>
          <Skeleton lines={4} />
        </Card>
      </>
    );
  }

  const ready = (papers ?? []).filter((p) => p.status === "ready");
  const working = (papers ?? []).filter((p) => p.status === "queued");
  const emb = embedderStatus(health.embedding_fallback);
  const llmOffline = health.llm === "offline";

  return (
    <>
      <PageHeader
        kicker="Overview"
        title="Dashboard"
        sub={
          isAdvanced
            ? "Advanced — papers, live engine detail, and per-paper provenance."
            : "Ask questions, get cited answers, and turn papers into code."
        }
      />
      {error && <div className="mb-6"><ErrorNote message={error} /></div>}

      <Stagger step={55} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <Card className="transition-shadow duration-200 hover:shadow-lift">
          <Stat label="Papers" value={ready.length} sub={`${working.length} processing`} />
        </Card>
        <Card className="transition-shadow duration-200 hover:shadow-lift">
          <Stat
            label="Embedder"
            value={
              <span className="flex items-center gap-2 text-xl">
                <StatusDot tone={emb.tone} />
                <span className={emb.tone === "warn" ? "text-warn" : "text-ok"}>
                  {emb.primary} ({emb.detail.toLowerCase()})
                </span>
              </span>
            }
            sub={health.embedding_model}
          />
        </Card>
        <Card className="transition-shadow duration-200 hover:shadow-lift">
          <Stat
            label="Reranker"
            value={<span className="text-xl">{health.reranker === "local" ? "Local" : health.reranker}</span>}
            sub={isAdvanced ? String((health.settings as Record<string, unknown>)["reranker_model"] ?? "") : "cross-encoder"}
          />
        </Card>
        <Card className="transition-shadow duration-200 hover:shadow-lift">
          <Stat
            label="LLM"
            value={
              <span className="flex items-center gap-2 text-xl">
                <StatusDot tone={llmOffline ? "warn" : "ok"} />
                {llmOffline ? "Offline" : health.llm}
              </span>
            }
            sub={llmOffline ? "extractive answering" : undefined}
          />
        </Card>
      </Stagger>

      {isAdvanced && (
        <div className="flex flex-wrap items-center gap-2 mb-8 animate-fadein">
          <Badge tone={emb.tone}>embedder: {emb.primary}</Badge>
          <Badge tone="neutral">private: {emb.detail}</Badge>
          <Badge tone={llmOffline ? "warn" : "ok"}>llm: {llmOffline ? "Offline" : health.llm}</Badge>
          <Badge tone="neutral">chunking: {health.chunk_strategy}</Badge>
          <Badge tone="neutral">parser: {health.parser}</Badge>
        </div>
      )}

      <div className="flex items-baseline justify-between mb-3">
        <h2 className="font-display text-xl">Papers</h2>
        <Link to="/papers" className="text-sm text-accent hover:underline">
          Open library →
        </Link>
      </div>
      {(papers ?? []).length === 0 ? (
        <Empty
          title="No papers ingested yet"
          hint={
            <>
              Upload a PDF in the <Link to="/papers" className="text-accent underline">library</Link>,
              or from the CLI: <code className="font-mono">researchrag ingest paper.pdf</code>
            </>
          }
        />
      ) : (
        <Stagger step={60} className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {(papers ?? []).map((p) => (
            <Link key={p.id} to={`/papers/${p.id}`} className="group focus:outline-none">
              <Card className="h-full transition-all duration-200 hover:shadow-lift hover:-translate-y-[1px] group-hover:border-accent/40 group-focus-visible:ring-2 group-focus-visible:ring-accent/60">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="font-display text-lg leading-snug break-words">
                      {p.title || p.filename}
                    </div>
                    <div className="text-xs text-ink-faint mt-1">
                      {p.page_count > 0 ? `${p.page_count} pages · ` : ""}
                      {fmtDate(p.created_at)}
                      {isAdvanced ? ` · ${p.parser ?? "pymupdf"} · ${p.id}` : ""}
                    </div>
                  </div>
                  <Badge
                    tone={
                      p.status === "ready" ? "ok" : p.status === "failed" ? "danger" : "warn"
                    }
                  >
                    {p.status}
                  </Badge>
                </div>
              </Card>
            </Link>
          ))}
        </Stagger>
      )}
    </>
  );
}
