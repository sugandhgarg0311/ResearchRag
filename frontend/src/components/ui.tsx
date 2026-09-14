import type { CSSProperties, ReactNode } from "react";
import { Link } from "react-router-dom";

// ---------------------------------------------------------------------------
// Layout primitives — editorial-minimal: hairline rules, no shadows except
// subtle card/lift, one accent color, serif display type for page titles.
// ---------------------------------------------------------------------------

export function Card({
  children,
  className = "",
  pad = true,
}: {
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <div
      className={`bg-card border border-line shadow-card rounded-sm ${
        pad ? "p-5" : ""
      } ${className}`}
    >
      {children}
    </div>
  );
}

export function PageHeader({
  kicker,
  title,
  sub,
  right,
}: {
  kicker?: string;
  title: string;
  sub?: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-8 animate-rise">
      <div className="min-w-0">
        {kicker && (
          <div className="text-[11px] uppercase tracking-[0.18em] text-ink-faint mb-2 font-medium">
            {kicker}
          </div>
        )}
        <h1 className="font-display text-2xl sm:text-3xl leading-tight tracking-tight break-words max-w-3xl">
          {title}
        </h1>
        {sub && <div className="text-sm text-ink-mute mt-2 max-w-2xl">{sub}</div>}
      </div>
      {right && <div className="shrink-0 flex flex-wrap items-center gap-2">{right}</div>}
    </div>
  );
}

export function Button({
  children,
  onClick,
  kind = "primary",
  disabled,
  busy,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  kind?: "primary" | "ghost" | "danger";
  disabled?: boolean;
  busy?: boolean;
  type?: "button" | "submit";
}) {
  const styles = {
    primary:
      "bg-ink text-paper hover:bg-ink-soft disabled:opacity-40 disabled:hover:bg-ink",
    ghost:
      "bg-transparent border border-line text-ink hover:bg-panel disabled:opacity-40",
    danger:
      "bg-transparent border border-danger/30 text-danger hover:bg-danger-soft disabled:opacity-40",
  } as const;
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || busy}
      className={`inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-sm transition-colors duration-150 cursor-pointer disabled:cursor-not-allowed ${styles[kind]}`}
    >
      {busy && <Spinner small />}
      {children}
    </button>
  );
}

export function Spinner({ small }: { small?: boolean }) {
  const s = small ? "h-3.5 w-3.5 border" : "h-5 w-5 border-2";
  return (
    <span
      aria-hidden
      className={`inline-block ${s} border-current border-t-transparent rounded-full animate-spin opacity-70`}
    />
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "ok" | "warn" | "danger" | "accent" | "infer" | "external";
}) {
  const tones = {
    neutral: "bg-panel text-ink-soft border-line",
    ok: "bg-ok-soft text-ok border-ok/20",
    warn: "bg-warn-soft text-warn border-warn/20",
    danger: "bg-danger-soft text-danger border-danger/20",
    accent: "bg-accent-soft text-accent border-accent/20",
    infer: "bg-infer-soft text-infer border-infer/20",
    external: "bg-external-soft text-external border-external/20",
  } as const;
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide rounded-sm border ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function sourceTone(source: string): "ok" | "warn" | "infer" | "external" | "neutral" {
  const s = source?.toUpperCase() ?? "";
  if (s === "EXPLICIT") return "ok";
  if (s === "INFERRED") return "infer";
  if (s === "EXTERNAL") return "external";
  if (s === "UNKNOWN") return "warn";
  return "neutral";
}

// ---------------------------------------------------------------------------
// Premium status + motion primitives (no new dependencies, no emoji).
// Simplified views show the plain-English word ("Offline"); advanced views
// add the private/engine detail ("Hashing", model ids, fallback flags).
// ---------------------------------------------------------------------------

export function StatusDot({ tone }: { tone: "ok" | "warn" | "danger" }) {
  const color = tone === "ok" ? "bg-ok" : tone === "warn" ? "bg-warn" : "bg-danger";
  const pulse = tone !== "ok" ? "dot-pulse" : "";
  return <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${color} ${pulse}`} />;
}

export function Skeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-2 animate-fadein" aria-hidden>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="skeleton h-3.5" style={{ width: `${92 - i * 12}%` }} />
      ))}
    </div>
  );
}

/** Staggered entrance wrapper: children rise in sequence. Respects reduced-motion via CSS. */
export function Stagger({
  children,
  step = 45,
  className = "",
}: {
  children: ReactNode;
  step?: number;
  className?: string;
}) {
  return (
    <div className={`stagger ${className}`} style={{ "--stagger": step } as CSSProperties}>
      {children}
    </div>
  );
}

/** Explicit status wording: primary word for everyone, private detail for advanced. */
export function embedderStatus(fallback: boolean): { primary: string; detail: string; tone: "warn" | "ok" } {
  return fallback
    ? { primary: "Offline", detail: "Hashing", tone: "warn" }
    : { primary: "Live", detail: "ONNX", tone: "ok" };
}

export function Empty({ title, hint }: { title: string; hint?: ReactNode }) {
  return (
    <div className="border border-dashed border-line rounded-sm py-14 px-6 text-center animate-fadein">
      <div className="font-display text-lg text-ink-soft">{title}</div>
      {hint && <div className="text-sm text-ink-faint mt-1.5 max-w-md mx-auto">{hint}</div>}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="border border-danger/30 bg-danger-soft rounded-sm px-4 py-3 text-sm text-danger animate-rise">
      {message}
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-10 justify-center text-ink-faint text-sm">
      <Spinner /> {label}
    </div>
  );
}

export function Table({
  head,
  children,
}: {
  head: Array<{ label: string; className?: string }>;
  children: ReactNode;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-line">
            {head.map((h) => (
              <th
                key={h.label}
                className={`text-left text-[11px] uppercase tracking-[0.14em] text-ink-faint font-medium py-2 pr-4 whitespace-nowrap ${
                  h.className ?? ""
                }`}
              >
                {h.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Td({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <td className={`py-2.5 pr-4 align-top border-b border-line/60 ${className}`}>{children}</td>;
}

export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: Array<{ id: string; label: string }>;
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="flex gap-1 border-b border-line mb-6" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={active === t.id}
          onClick={() => onChange(t.id)}
          className={`px-4 py-2.5 text-sm -mb-px border-b-2 transition-colors cursor-pointer ${
            active === t.id
              ? "border-accent text-ink font-medium"
              : "border-transparent text-ink-faint hover:text-ink-soft"
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium">
        {label}
      </div>
      <div className="font-display text-2xl mt-1">{value}</div>
      {sub && <div className="text-xs text-ink-faint mt-0.5">{sub}</div>}
    </div>
  );
}

export function MetricBar({
  label,
  value,
  max = 1,
  digits = 4,
}: {
  label: string;
  value: number | null | undefined;
  max?: number;
  digits?: number;
}) {
  const v = value ?? 0;
  const pct = max > 0 ? Math.min(100, (v / max) * 100) : 0;
  return (
    <div className="min-w-[10rem]">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-xs text-ink-soft">{label}</span>
        <span className="font-mono text-sm tabular-nums">
          {value == null ? "—" : value.toFixed(digits)}
        </span>
      </div>
      <div className="h-1 bg-panel rounded-full mt-1.5 overflow-hidden border border-line/60">
        <div className="h-full bg-accent transition-all duration-300" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function Code({ children }: { children: string }) {
  return (
    <pre className="bg-panel border border-line rounded-sm p-4 overflow-x-auto text-[12.5px] leading-relaxed font-mono whitespace-pre text-ink-soft">
      {children}
    </pre>
  );
}

export function Prose({ children }: { children: string }) {
  return <div className="prose-chunk text-ink">{children}</div>;
}

export function Breadcrumbs({ items }: { items: Array<{ label: string; to?: string }> }) {
  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 text-xs text-ink-faint mb-4">
      {items.map((it, i) => (
        <span key={i} className="flex items-center gap-1.5">
          {i > 0 && <span aria-hidden>/</span>}
          {it.to ? (
            <Link to={it.to} className="hover:text-ink transition-colors">
              {it.label}
            </Link>
          ) : (
            <span className="text-ink-soft">{it.label}</span>
          )}
        </span>
      ))}
    </nav>
  );
}
