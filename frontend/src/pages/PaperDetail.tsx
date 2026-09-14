import { useEffect, useMemo, useState } from "react";
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
  Table,
  Tabs,
  Td,
} from "../components/ui";
import type { Chunk, StructureSection } from "../types";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "structure", label: "Structure" },
  { id: "chunks", label: "Chunks" },
  { id: "pages", label: "Pages" },
];

export default function PaperDetail() {
  const { paperId = "" } = useParams();
  const [tab, setTab] = useState("overview");
  const paper = useAsync(() => api.getPaper(paperId), [paperId]);
  const structure = useAsync(() => api.structure(paperId), [paperId]);
  const chunks = useAsync(() => api.chunks(paperId, "child"), [paperId]);

  if (paper.error) return <ErrorNote message={paper.error} />;
  if (!paper.data) return <Loading label="Loading paper…" />;

  const p = paper.data;
  const children = (chunks.data ?? []).filter((c) => c.kind === "child");
  const parents = (chunks.data ?? []).filter((c) => c.kind === "parent");
  const crossPage = children.filter((c) => c.page_end > c.page_start).length;

  return (
    <>
      <Breadcrumbs items={[{ label: "Papers", to: "/papers" }, { label: p.id }]} />
      <PageHeader
        kicker={`${p.status} · ${p.parser ?? "pymupdf"} · ${fmtDate(p.created_at)}`}
        title={p.title || p.filename}
        sub={`${p.page_count} pages · ${children.length} child chunks · ${parents.length} parents · ${crossPage} cross-page`}
        right={
          <>
            <Link
              to={`/papers/${p.id}/ask`}
              className="px-4 py-2 text-sm font-medium bg-ink text-paper rounded-sm hover:bg-ink-soft transition-colors"
            >
              Ask
            </Link>
            <Link
              to={`/papers/${p.id}/retrieve`}
              className="px-4 py-2 text-sm font-medium border border-line rounded-sm hover:bg-panel transition-colors"
            >
              Retrieve
            </Link>
            <Link
              to={`/papers/${p.id}/specs`}
              className="px-4 py-2 text-sm font-medium border border-line rounded-sm hover:bg-panel transition-colors"
            >
              Specs
            </Link>
          </>
        }
      />
      <Tabs tabs={TABS} active={tab} onChange={setTab} />

      {tab === "overview" && <Overview p={p} />}
      {tab === "structure" && (
        <StructureView
          loading={structure.loading}
          error={structure.error}
          sections={structure.data?.sections ?? []}
          paperId={p.id}
          figures={(structure.data?.pages ?? []).flatMap((pg) =>
            pg.blocks
              .filter((b) => b.type === "figure" && b.has_image)
              .map((b) => ({ id: b.id, caption: b.caption, page: pg.number }))
          )}
        />
      )}
      {tab === "chunks" && (
        <ChunksView
          loading={chunks.loading || paper.loading}
          error={chunks.error}
          chunks={children}
        />
      )}
      {tab === "pages" && (
        <PagesView paperId={p.id} pageCount={p.page_count} />
      )}
    </>
  );
}

function Overview({ p }: { p: Awaited<ReturnType<typeof api.getPaper>> }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 animate-rise">
      <Card>
        <h3 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mb-4">
          Pipeline actions
        </h3>
        <ul className="space-y-2.5 text-sm">
          {[
            { to: `/papers/${p.id}/ask`, label: "Grounded question answering" },
            { to: `/papers/${p.id}/retrieve`, label: "Hybrid retrieval debug" },
            { to: `/papers/${p.id}/specs`, label: "Implementation specification" },
            { to: `/papers/${p.id}/implementations`, label: "Code generation" },
            { to: `/papers/${p.id}/reproduction`, label: "Reproduction comparison" },
            { to: `/papers/${p.id}/evaluations`, label: "Evaluation" },
            { to: `/papers/${p.id}/experiments`, label: "Retrieval experiments" },
          ].map((l) => (
            <li key={l.to}>
              <Link to={l.to} className="text-accent hover:underline">
                {l.label} →
              </Link>
            </li>
          ))}
        </ul>
      </Card>
      <Card>
        <h3 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mb-4">
          Paper
        </h3>
        <dl className="space-y-2.5 text-sm">
          <Row k="id" v={<code className="font-mono text-xs">{p.id}</code>} />
          <Row k="file" v={p.filename} />
          <Row k="pages" v={String(p.page_count)} />
          <Row k="created" v={fmtDate(p.created_at)} />
          {p.authors && p.authors.length > 0 && (
            <Row k="authors" v={p.authors.join(", ")} />
          )}
        </dl>
      </Card>
      <Card>
        <h3 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mb-4">
          Recent jobs
        </h3>
        {(p.jobs ?? []).length === 0 ? (
          <div className="text-sm text-ink-faint">No jobs recorded.</div>
        ) : (
          <ul className="space-y-2 text-sm">
            {(p.jobs ?? []).map((j) => (
              <li key={j.id} className="flex items-center justify-between gap-3">
                <span className="text-ink-soft">
                  {j.kind} · {fmtDate(j.created_at)}
                </span>
                <Badge tone={j.status === "done" ? "ok" : j.status === "failed" ? "danger" : "warn"}>
                  {j.status}
                </Badge>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-ink-faint shrink-0">{k}</dt>
      <dd className="text-right min-w-0 break-words">{v}</dd>
    </div>
  );
}

function StructureView({
  loading,
  error,
  sections,
  paperId,
  figures,
}: {
  loading: boolean;
  error: string | null;
  sections: StructureSection[];
  paperId: string;
  figures: Array<{ id: string; caption: string | null; page: number }>;
}) {
  if (loading) return <Loading label="Loading structure…" />;
  if (error) return <ErrorNote message={error} />;
  if (sections.length === 0)
    return <Empty title="No headings found" hint="The parser found no section headings in this paper." />;
  return (
    <div className="space-y-6 animate-rise">
      {figures.length > 0 && (
        <Card>
          <h3 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mb-3">
            Figures ({figures.length})
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {figures.map((f) => (
              <figure key={f.id} className="border border-line rounded-sm overflow-hidden bg-panel/40 group">
                <img
                  src={`/api/papers/${paperId}/blocks/${f.id}/image`}
                  alt={f.caption ?? `Figure on page ${f.page}`}
                  loading="lazy"
                  className="w-full h-28 object-cover transition-transform duration-200 group-hover:scale-[1.02]"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
                <figcaption className="px-2.5 py-2 text-[11px] text-ink-mute leading-snug line-clamp-2">
                  {f.caption ?? `p.${f.page}`}
                </figcaption>
              </figure>
            ))}
          </div>
        </Card>
      )}
      <Card pad={false}>
        <ul className="divide-y divide-line/70">
          {sections.map((s) => (
            <li
              key={`${s.id}-${s.block_id}`}
              className="flex items-baseline gap-3 px-5 py-2.5 hover:bg-panel/60 transition-colors"
              style={{ paddingLeft: `${(s.level - 1) * 1.25 + 1.25}rem` }}
            >
              <span className="font-mono text-[11px] text-ink-faint w-10 shrink-0">§{s.id}</span>
              <span className="text-sm font-medium truncate">{s.title}</span>
              <span className="ml-auto text-[11px] text-ink-faint shrink-0">
                {s.path.join(" › ")} · p.{s.page}
              </span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}

function ChunksView({
  loading,
  error,
  chunks,
}: {
  loading: boolean;
  error: string | null;
  chunks: Chunk[];
}) {
  const [open, setOpen] = useState<Chunk | null>(null);
  if (loading) return <Loading label="Loading chunks…" />;
  if (error) return <ErrorNote message={error} />;
  return (
    <div className="animate-rise">
      {open && (
        <Card className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <code className="font-mono text-[11px] text-ink-faint">{open.id}</code>
            <button onClick={() => setOpen(null)} className="text-xs text-ink-faint hover:text-ink cursor-pointer">
              close
            </button>
          </div>
          <div className="flex gap-2 mb-3">
            <Badge tone="neutral">{open.kind}</Badge>
            <Badge tone="neutral">{open.page_start === open.page_end ? `p.${open.page_start}` : `p.${open.page_start}–${open.page_end}`}</Badge>
            <Badge tone="neutral">{open.chunk_type}</Badge>
            <Badge tone="accent">{open.token_count} tokens</Badge>
          </div>
          {open.chunk_type === "formula" ? (
            <div className="border border-line rounded-sm bg-panel/60 p-4">
              <div className="math-scroll font-math text-[17px] leading-relaxed text-ink">{open.text}</div>
              <button
                onClick={() => void navigator.clipboard?.writeText(open.text)}
                className="mt-2.5 font-mono text-[11px] text-accent hover:underline cursor-pointer"
              >
                copy equation
              </button>
            </div>
          ) : (
            <p className="text-sm leading-relaxed whitespace-pre-wrap max-h-96 overflow-y-auto">{open.text}</p>
          )}
        </Card>
      )}
      <Card pad={false}>
        <Table
          head={[
            { label: "Chunk" },
            { label: "Section" },
            { label: "Pages" },
            { label: "Tokens" },
            { label: "Type" },
          ]}
        >
          {chunks.slice(0, 200).map((c) => (
            <tr
              key={c.id}
              onClick={() => setOpen(c)}
              className="cursor-pointer hover:bg-panel/50 transition-colors"
            >
              <Td className="max-w-[24rem]">
                <span className="text-ink-soft line-clamp-1">{c.text}</span>
              </Td>
              <Td>
                <span className="text-xs text-ink-faint">{c.section_path.join(" › ") || "—"}</span>
              </Td>
              <Td>
                <span className="font-mono text-xs">
                  {c.page_start === c.page_end ? `p.${c.page_start}` : `p.${c.page_start}–${c.page_end}`}
                </span>
              </Td>
              <Td>
                <span className="font-mono text-xs">{c.token_count}</span>
              </Td>
              <Td>
                <span className="text-xs text-ink-faint">{c.chunk_type}</span>
              </Td>
            </tr>
          ))}
        </Table>
        {chunks.length > 200 && (
          <div className="px-5 py-3 text-xs text-ink-faint border-t border-line">
            Showing first 200 of {chunks.length} child chunks.
          </div>
        )}
      </Card>
    </div>
  );
}

function PagesView({ paperId, pageCount }: { paperId: string; pageCount: number }) {
  const [n, setN] = useState(1);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [retry, setRetry] = useState(0);
  const nums = useMemo(() => Array.from({ length: Math.min(pageCount, 200) }, (_, i) => i + 1), [pageCount]);
  const src = `/api/papers/${paperId}/pages/${n}/image?dpi=110`;

  // Reset on page/paper change and warm the neighbours via JS Image
  // preloading. (The previous hidden-<img>-in-display:none preloader never
  // fired — lazy images inside display:none are never fetched, so the main
  // image stayed on "Rendering…" forever. Pages beyond the first 12 were
  // never preloaded at all.)
  useEffect(() => {
    setStatus("loading");
    [n - 1, n + 1, n + 2].forEach((i) => {
      if (i >= 1 && i <= nums.length && i !== n) {
        const im = new Image();
        im.src = `/api/papers/${paperId}/pages/${i}/image?dpi=110`;
      }
    });
  }, [n, paperId, nums.length]);

  if (pageCount <= 0) return <Empty title="No page count recorded" />;
  return (
    <div className="grid grid-cols-1 md:grid-cols-[1fr_9rem] gap-6 animate-rise">
      <Card className="flex flex-col items-center justify-center bg-panel/40 min-h-[40vh] gap-3">
        {status === "loading" && <Loading label={`Rendering page ${n}…`} />}
        {status === "error" && (
          <div className="w-full">
            <ErrorNote message={`Could not render page ${n}. The API may be offline or the page image cache may have been cleared.`} />
            <div className="flex gap-2 mt-3">
              <button
                onClick={() => {
                  setRetry((r) => r + 1);
                  setStatus("loading");
                }}
                className="px-3 py-1.5 text-xs border border-line rounded-sm bg-card text-ink-soft hover:bg-panel hover:text-ink transition-colors cursor-pointer"
              >
                Retry
              </button>
              <a
                href={src}
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 text-xs border border-line rounded-sm bg-card text-ink-soft hover:bg-panel hover:text-ink transition-colors"
              >
                Open image directly
              </a>
            </div>
          </div>
        )}
        <img
          key={`${src}#${retry}`}
          src={src}
          alt={`Page ${n} of ${pageCount}`}
          onLoad={() => setStatus("ready")}
          onError={() => setStatus("error")}
          className={`max-h-[75vh] w-auto shadow-lift animate-fadein ${status === "ready" ? "" : "hidden"}`}
        />
        {status === "ready" && (
          <div className="text-[11px] text-ink-faint font-mono">
            p.{n} / {pageCount}
          </div>
        )}
      </Card>
      <div className="flex md:flex-col gap-1 overflow-x-auto md:overflow-visible md:max-h-[75vh] md:overflow-y-auto pr-1 pb-1" role="listbox" aria-label="Pages">
        {nums.map((i) => (
          <button
            key={i}
            onClick={() => setN(i)}
            aria-current={i === n}
            aria-label={`Page ${i}`}
            className={`shrink-0 md:w-full text-left px-3 py-2 text-sm rounded-sm border transition-all duration-150 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 ${
              i === n
                ? "border-accent bg-accent-soft text-accent font-medium"
                : "border-line hover:bg-panel hover:translate-x-[1px] text-ink-soft"
            }`}
          >
            p.{i}
          </button>
        ))}
      </div>
    </div>
  );
}
