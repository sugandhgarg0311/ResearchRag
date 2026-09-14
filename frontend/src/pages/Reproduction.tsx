import { useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import { useAsync, fmtDate, fmtNum } from "../hooks";
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

interface Draft {
  metric: string;
  paperValue: string;
  ourValue: string;
  notes: string;
}

export default function Reproduction() {
  const { paperId = "" } = useParams();
  const paper = useAsync(() => api.getPaper(paperId), [paperId]);
  const { data: records, loading, error, reload } = useAsync(
    () => api.reproduction(paperId),
    [paperId]
  );
  const [rows, setRows] = useState<Draft[]>([
    { metric: "", paperValue: "", ourValue: "", notes: "" },
  ]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [ok, setOk] = useState(false);

  const grouped = useMemo(() => {
    const m = new Map<string, NonNullable<typeof records>>();
    for (const r of records ?? []) {
      const arr = m.get(r.metric) ?? [];
      arr.push(r);
      m.set(r.metric, arr);
    }
    return [...m.entries()];
  }, [records]);

  const update = (i: number, patch: Partial<Draft>) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)));

  const submit = async () => {
    const payload = rows
      .filter((r) => r.metric.trim())
      .map((r) => ({
        metric: r.metric.trim(),
        paper_value: r.paperValue.trim() === "" ? null : Number(r.paperValue),
        our_value: r.ourValue.trim() === "" ? null : Number(r.ourValue),
        notes: r.notes.trim() || null,
      }));
    if (payload.length === 0) return;
    setBusy(true);
    setErr(null);
    setOk(false);
    try {
      await api.addReproduction(paperId, payload);
      setRows([{ metric: "", paperValue: "", ourValue: "", notes: "" }]);
      setOk(true);
      reload();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Breadcrumbs
        items={[
          { label: "Papers", to: "/papers" },
          { label: paper.data?.title || paperId, to: `/papers/${paperId}` },
          { label: "Reproduction" },
        ]}
      />
      <PageHeader
        kicker="Reproduction"
        title="Paper vs. our runs"
        sub="Every metric is compared against the paper's reported value. Where the paper reports nothing, the gap is shown as a gap — never filled in."
      />

      {(records ?? []).length > 0 && (
        <div className="space-y-4 mb-8 animate-rise">
          {grouped.map(([metric, rs]) => (
            <Card key={metric}>
              <h3 className="font-display text-lg mb-3">{metric}</h3>
              <Table
                head={[
                  { label: "Paper reports" },
                  { label: "Our run" },
                  { label: "Difference" },
                  { label: "Notes" },
                  { label: "Recorded" },
                ]}
              >
                {rs.map((r) => (
                  <tr key={r.id}>
                    <Td>
                      {r.paper_value != null ? (
                        <span className="font-mono">{fmtNum(r.paper_value, 2)}</span>
                      ) : (
                        <Badge tone="warn">not reported</Badge>
                      )}
                    </Td>
                    <Td>
                      {r.our_value != null ? (
                        <span className="font-mono">{fmtNum(r.our_value, 2)}</span>
                      ) : (
                        <span className="text-ink-faint">—</span>
                      )}
                    </Td>
                    <Td>
                      {r.difference != null ? (
                        <span className={`font-mono ${r.difference < 0 ? "text-danger" : "text-ok"}`}>
                          {r.difference >= 0 ? "+" : ""}
                          {fmtNum(r.difference, 2)}
                        </span>
                      ) : (
                        <span className="text-ink-faint">—</span>
                      )}
                    </Td>
                    <Td>
                      <span className="text-xs text-ink-soft">{r.notes || "—"}</span>
                    </Td>
                    <Td>
                      <span className="text-xs text-ink-faint">{fmtDate(r.created_at)}</span>
                    </Td>
                  </tr>
                ))}
              </Table>
            </Card>
          ))}
        </div>
      )}

      <Card className="animate-rise">
        <h3 className="text-[11px] uppercase tracking-[0.16em] text-ink-faint font-medium mb-1">
          Record a comparison
        </h3>
        <p className="text-xs text-ink-faint mb-4">
          Run metrics are recorded automatically by sandbox runs that emit{" "}
          <code className="font-mono">METRICS: {"{…}"}</code> lines; this form records paper-side
          values (e.g. from Table 1) and ad-hoc comparisons.
        </p>
        {err && <div className="mb-3"><ErrorNote message={err} /></div>}
        {ok && (
          <div className="mb-3 border border-ok/30 bg-ok-soft rounded-sm px-3 py-2 text-sm text-ok animate-rise">
            Recorded.
          </div>
        )}
        <div className="space-y-2">
          {rows.map((r, i) => (
            <div key={i} className="grid grid-cols-[12rem_8rem_8rem_1fr_auto] gap-2 items-center">
              <input
                value={r.metric}
                onChange={(e) => update(i, { metric: e.target.value })}
                placeholder="metric (e.g. em_nq)"
                className="border border-line rounded-sm px-2.5 py-1.5 text-sm focus:outline-none focus:border-accent"
              />
              <input
                value={r.paperValue}
                onChange={(e) => update(i, { paperValue: e.target.value })}
                placeholder="paper value"
                inputMode="decimal"
                className="border border-line rounded-sm px-2.5 py-1.5 text-sm font-mono focus:outline-none focus:border-accent"
              />
              <input
                value={r.ourValue}
                onChange={(e) => update(i, { ourValue: e.target.value })}
                placeholder="our value"
                inputMode="decimal"
                className="border border-line rounded-sm px-2.5 py-1.5 text-sm font-mono focus:outline-none focus:border-accent"
              />
              <input
                value={r.notes}
                onChange={(e) => update(i, { notes: e.target.value })}
                placeholder="notes (dataset, seed, …)"
                className="border border-line rounded-sm px-2.5 py-1.5 text-sm focus:outline-none focus:border-accent"
              />
              {rows.length > 1 && (
                <button
                  onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}
                  className="text-xs text-ink-faint hover:text-danger cursor-pointer"
                >
                  remove
                </button>
              )}
            </div>
          ))}
        </div>
        <div className="flex items-center gap-2 mt-4">
          <Button kind="ghost" onClick={() => setRows((rs) => [...rs, { metric: "", paperValue: "", ourValue: "", notes: "" }])}>
            Add row
          </Button>
          <Button onClick={submit} busy={busy} disabled={!rows.some((r) => r.metric.trim())}>
            Record
          </Button>
        </div>
      </Card>

      {loading && <div className="mt-4"><Loading label="Loading records…" /></div>}
      {error && <div className="mt-4"><ErrorNote message={error} /></div>}
      {!loading && !error && (records ?? []).length === 0 && (
        <div className="mt-6">
          <Empty
            title="No reproduction records yet"
            hint="Run a generated project (sandbox) — METRICS lines are compared against paper values automatically — or record values manually above."
          />
        </div>
      )}
    </>
  );
}
