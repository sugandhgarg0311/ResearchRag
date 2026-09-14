import { useState } from "react";
import { api } from "../api";
import { useAsync } from "../hooks";
import { useViewMode } from "../view";
import {
  Badge,
  Card,
  Code,
  ErrorNote,
  PageHeader,
  Skeleton,
  Stagger,
  Stat,
  StatusDot,
  embedderStatus,
} from "../components/ui";

export default function HealthPage() {
  const { data: h, loading, error } = useAsync(() => api.health(), []);
  const [showConfig, setShowConfig] = useState(false);
  const [copied, setCopied] = useState(false);
  const { isAdvanced } = useViewMode();

  if (error) return <ErrorNote message={error} />;
  if (loading || !h) {
    return (
      <>
        <PageHeader kicker="System" title="Pipeline components" sub="Checking what is actually running…" />
        <Card>
          <Skeleton lines={4} />
        </Card>
      </>
    );
  }

  const emb = embedderStatus(h.embedding_fallback);
  const llmOffline = h.llm === "offline";
  const rerankLocal = h.reranker === "local";

  const copyConfig = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(h.settings, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable — ignore */
    }
  };

  return (
    <>
      <PageHeader
        kicker="System"
        title="Pipeline components"
        sub={
          isAdvanced
            ? "Full developer view — active engine, private fallback detail, and the complete configuration."
            : "Plain-English status. Switch to Advanced for engine detail and configuration."
        }
      />

      <Stagger step={55} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <Card className="transition-shadow duration-200 hover:shadow-lift">
          <Stat
            label="Embedder"
            value={
              <span className="flex items-center gap-2 text-xl">
                <StatusDot tone={emb.tone} />
                <span className={emb.tone === "warn" ? "text-warn" : "text-ok"}>{emb.primary}</span>
              </span>
            }
            sub={`Private: ${emb.detail} · ${h.embedding_model}`}
          />
          {!isAdvanced && (
            <div className="mt-2 text-xs text-ink-mute">
              {h.embedding_fallback
                ? "No model download — answers still work, quality is reduced."
                : "Dense vectors are live."}
            </div>
          )}
        </Card>
        <Card className="transition-shadow duration-200 hover:shadow-lift">
          <Stat
            label="Reranker"
            value={
              <span className="flex items-center gap-2 text-xl">
                <StatusDot tone="ok" />
                {rerankLocal ? "Local" : h.reranker}
              </span>
            }
            sub={isAdvanced ? String((h.settings as Record<string, unknown>)["reranker_model"] ?? "") : "ONNX cross-encoder"}
          />
        </Card>
        <Card className="transition-shadow duration-200 hover:shadow-lift">
          <Stat
            label="LLM"
            value={
              <span className="flex items-center gap-2 text-xl">
                <StatusDot tone={llmOffline ? "warn" : "ok"} />
                <span className={llmOffline ? "text-warn" : ""}>{llmOffline ? "Offline" : h.llm}</span>
              </span>
            }
            sub={llmOffline ? "extractive answering (no key needed)" : "generative answering"}
          />
        </Card>
        <Card className="transition-shadow duration-200 hover:shadow-lift">
          <Stat label="Parser" value={<span className="text-xl">{h.parser}</span>} sub={`chunking: ${h.chunk_strategy}`} />
        </Card>
      </Stagger>

      <div className="flex flex-wrap items-center gap-3 mb-6 animate-fadein">
        <Badge tone="ok">status: {h.status}</Badge>
        <Badge tone="neutral">v{h.version}</Badge>
        <Badge tone={emb.tone}>embedder: {emb.primary}</Badge>
        {isAdvanced && <Badge tone="neutral">private: {emb.detail}</Badge>}
        <Badge tone={llmOffline ? "warn" : "ok"}>llm: {llmOffline ? "Offline" : h.llm}</Badge>
        {h.embedding_fallback && (
          <span className="text-xs text-warn">
            Dense model weights are unavailable — the deterministic hashing embedder is in use.
            Allow Hugging Face egress once and re-ingest to switch.
          </span>
        )}
      </div>

      {isAdvanced ? (
        <Card>
          <div className="flex items-center justify-between gap-3">
            <button
              onClick={() => setShowConfig((s) => !s)}
              aria-expanded={showConfig}
              className="text-sm text-accent hover:underline cursor-pointer"
            >
              {showConfig ? "hide configuration" : "show configuration"}
            </button>
            <button
              onClick={copyConfig}
              className="text-xs border border-line rounded-sm px-2.5 py-1.5 text-ink-soft hover:bg-panel hover:text-ink transition-colors cursor-pointer"
            >
              {copied ? "Copied" : "Copy JSON"}
            </button>
          </div>
          {showConfig && (
            <div className="mt-4 animate-fadein">
              <Code>{JSON.stringify(h.settings, null, 2)}</Code>
            </div>
          )}
        </Card>
      ) : (
        <Card>
          <div className="text-sm text-ink-soft leading-relaxed">
            Everything runs locally. The embedder is <strong>Offline</strong> (private engine:{" "}
            <strong>Hashing</strong>), the reranker is <strong>Local</strong>, and the LLM is{" "}
            <strong>Offline</strong> — so answers are extractive and fully cited. No keys required.
          </div>
        </Card>
      )}
    </>
  );
}
