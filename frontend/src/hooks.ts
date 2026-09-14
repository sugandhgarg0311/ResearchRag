import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Job } from "./types";

/** Run an async fetcher, track loading/error, allow re-run. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fnRef
      .current()
      .then((d) => alive && (setData(d), setLoading(false)))
      .catch((e: Error) => alive && (setError(e.message), setLoading(false)));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, loading, error, reload };
}

/** Poll a background job until it reaches a terminal state. */
export function useJob(jobId: string | null) {
  const [job, setJob] = useState<Job | null>(null);
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      try {
        const j = await api.getJob(jobId);
        if (!alive) return;
        setJob(j);
        if ((j.status === "done" || j.status === "failed") && timer) return;
        timer = setTimeout(tick, 1200);
      } catch {
        if (timer) timer = setTimeout(tick, 2500);
      }
    };
    void tick();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);
  return job;
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`;
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v == null) return "—";
  return v.toFixed(digits);
}
