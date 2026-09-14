import { NavLink, Outlet } from "react-router-dom";
import type { CSSProperties, ReactNode } from "react";
import { useAsync } from "../hooks";
import { api } from "../api";
import { ViewToggle, useViewMode } from "../view";
import { ThemeToggle } from "../theme";
import { StatusDot, embedderStatus } from "./ui";

const NAV: Array<{ to: string; label: string; end?: boolean; hint: string; icon: ReactNode }> = [
  {
    to: "/",
    label: "Dashboard",
    end: true,
    hint: "Papers and pipeline status at a glance",
    icon: (
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
        <rect x="1.5" y="1.5" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.3" />
        <rect x="9" y="1.5" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.3" />
        <rect x="1.5" y="9" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.3" />
        <rect x="9" y="9" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.3" />
      </svg>
    ),
  },
  {
    to: "/papers",
    label: "Papers",
    hint: "Upload, parse and inspect research PDFs",
    icon: (
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
        <path d="M4 1.5h5.5L13 5v9.5H4z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
        <path d="M9.5 1.5V5H13" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: "/chat",
    label: "Chat",
    hint: "Chat with any ingested paper",
    icon: (
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
        <path d="M2 3.5h12v7H8l-3.5 3v-3H2z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
        <path d="M5 6.5h6M5 8.5h4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    to: "/jobs",
    label: "Jobs",
    hint: "Background ingest, spec, run and eval jobs",
    icon: (
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
        <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.3" />
        <path d="M8 4.5V8l2.5 1.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    to: "/health",
    label: "System",
    hint: "Embedder, reranker, LLM and configuration",
    icon: (
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
        <path d="M2 9.5 5.5 5l3 3 3-3L14 8" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M2 13.5h12" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      </svg>
    ),
  },
];

export default function Layout() {
  const { data: health } = useAsync(() => api.health(), []);
  const { isAdvanced } = useViewMode();
  const emb = health ? embedderStatus(health.embedding_fallback) : null;

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-paper">
      <aside className="w-full md:w-60 md:shrink-0 border-b md:border-b-0 md:border-r border-line bg-card/90 backdrop-blur flex flex-col md:sticky md:top-0 md:h-screen">
        <div className="px-5 pt-5 md:pt-6 pb-4 md:pb-5 border-b border-line">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="font-display text-xl tracking-tight leading-none">
                Research<span className="text-accent">RAG</span>
              </div>
              <div className="text-[11px] text-ink-faint mt-1.5 tracking-wide">
                evidence-first paper → code
              </div>
            </div>
            <ThemeToggle />
          </div>
          <div className="mt-4">
            <ViewToggle />
            <div className="mt-1.5 text-[11px] text-ink-faint">
              {isAdvanced ? "Advanced — full pipeline detail" : "Simplified — normal users"}
            </div>
          </div>
        </div>
        <nav aria-label="Primary" className="flex md:flex-col gap-0.5 px-3 py-3 md:py-4 overflow-x-auto md:overflow-visible md:flex-1 stagger" style={{ "--stagger": 35 } as CSSProperties}>
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              title={n.hint}
              className={({ isActive }) =>
                `group relative flex items-center gap-2.5 px-3 py-2 rounded-sm text-sm whitespace-nowrap transition-all duration-150 focus:outline-none ${
                  isActive
                    ? "bg-panel text-ink font-medium"
                    : "text-ink-mute hover:text-ink hover:bg-panel/60 hover:translate-x-[1px]"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span
                    aria-hidden
                    className={`absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full bg-accent transition-opacity duration-150 ${
                      isActive ? "opacity-100" : "opacity-0 group-hover:opacity-40"
                    }`}
                  />
                  <span className={isActive ? "text-accent" : "text-ink-faint group-hover:text-ink-soft transition-colors"}>
                    {n.icon}
                  </span>
                  {n.label}
                </>
              )}
            </NavLink>
          ))}
        </nav>
        <div className="hidden md:block px-5 py-4 border-t border-line text-[11px] text-ink-faint leading-relaxed">
          {health && emb ? (
            <>
              <div className="flex items-center gap-1.5 text-ink-soft">
                <StatusDot tone={emb.tone} />
                <span className="font-medium">
                  Embedder: {emb.primary}
                  <span className="text-ink-faint"> · {emb.detail}</span>
                </span>
              </div>
              <div className="mt-1">
                Reranker: {health.reranker === "local" ? "Local" : health.reranker} · LLM:{" "}
                {health.llm === "offline" ? "Offline" : health.llm}
              </div>
              {isAdvanced && (
                <div className="mt-1.5 font-mono text-[10.5px] text-ink-faint break-all animate-fadein">
                  {health.embedding_model}
                  {health.embedding_fallback ? " · fallback=true" : ""}
                </div>
              )}
            </>
          ) : (
            "API offline — start the server"
          )}
        </div>
      </aside>
      <main className="flex-1 min-w-0">
        <div className="max-w-content mx-auto px-4 sm:px-8 py-6 sm:py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
