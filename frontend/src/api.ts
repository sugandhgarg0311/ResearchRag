// Minimal typed API client. All calls use relative URLs so the same bundle
// works against the Vite dev proxy (/:5173 → :8000) and in production where
// the API serves the built frontend itself.

import type {
  Answer,
  Chunk,
  Experiment,
  EvalRun,
  Health,
  Implementation,
  ImplementationSummary,
  Job,
  Paper,
  PaperStructure,
  ReproductionRecord,
  RetrievalConfig,
  RetrievalResult,
  RunRecord,
  Spec,
  SpecSummary,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const get = <T>(path: string) => request<T>(path);
const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
const patch = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "PATCH", body: JSON.stringify(body ?? {}) });
const del = <T>(path: string) => request<T>(path, { method: "DELETE" });

export const api = {
  // health
  health: () => get<Health>("/api/health"),

  // papers
  listPapers: () => get<{ papers: Paper[] }>("/api/papers").then((r) => r.papers),
  getPaper: (id: string) => get<Paper>(`/api/papers/${id}`),
  structure: (id: string) => get<PaperStructure>(`/api/papers/${id}/structure`),
  chunks: (id: string, kind?: string) =>
    get<{ chunks: Chunk[] }>(`/api/papers/${id}/chunks${kind ? `?kind=${kind}` : ""}`).then(
      (r) => r.chunks
    ),
  uploadPaper: async (file: File): Promise<{ paper_id: string; job_id: string }> => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/papers", { method: "POST", body: fd });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new ApiError(res.status, typeof body.detail === "string" ? body.detail : "Upload failed");
    }
    return res.json();
  },
  deletePaper: (id: string) => del<void>(`/api/papers/${id}`),

  // search
  chat: (paperId: string, question: string, config?: RetrievalConfig) =>
    post<Answer>(`/api/papers/${paperId}/chat`, { question, config: config ?? null }),
  retrieve: (paperId: string, query: string, config?: RetrievalConfig) =>
    post<RetrievalResult>(`/api/papers/${paperId}/retrieve`, { query, config: config ?? null }),

  // specs
  listSpecs: (paperId: string) =>
    get<{ specs: SpecSummary[] }>(`/api/papers/${paperId}/specs`).then((r) => r.specs),
  getSpec: (specId: string) => get<Spec>(`/api/specs/${specId}`),
  generateSpec: (paperId: string) => post<{ job_id: string }>(`/api/papers/${paperId}/specs`, {}),
  setRequirementStatus: (specId: string, reqId: string, status: string, notes?: string) =>
    patch<{ ok: boolean }>(
      `/api/specs/${specId}/requirements/${reqId}?status=${encodeURIComponent(status)}`,
      notes ? { notes } : undefined
    ),

  // implementations
  listImplementations: (paperId: string) =>
    get<{ implementations: ImplementationSummary[] }>(
      `/api/papers/${paperId}/implementations`
    ).then((r) => r.implementations),
  getImplementation: (implId: string) => get<Implementation>(`/api/implementations/${implId}`),
  getFile: (implId: string, path: string) =>
    get<{ path: string; content: string; kind: string }>(
      `/api/implementations/${implId}/files/${encodeURIComponent(path)}`
    ),
  generateImplementation: (paperId: string, specId?: string) =>
    post<{ job_id: string }>(`/api/papers/${paperId}/implementations`, {
      spec_id: specId ?? null,
      llm: true,
    }),
  listRuns: (paperId: string) =>
    get<{ runs: RunRecord[] }>(`/api/papers/${paperId}/runs`).then((r) => r.runs),
  runImplementation: (implId: string, kind: string) =>
    post<{ job_id: string }>(`/api/implementations/${implId}/runs`, { kind }),

  // reproduction
  reproduction: (paperId: string) =>
    get<{ records: ReproductionRecord[] }>(`/api/papers/${paperId}/reproduction`).then(
      (r) => r.records
    ),
  addReproduction: (paperId: string, metrics: Array<Record<string, unknown>>) =>
    post<{ created: Array<{ id: string; metric: string }> }>(
      `/api/papers/${paperId}/reproduction`,
      metrics
    ),

  // evaluation
  listDatasets: (paperId: string) =>
    get<{ datasets: string[] }>(`/api/papers/${paperId}/eval-datasets`).then((r) => r.datasets),
  listEvaluations: (paperId: string) =>
    get<{ evaluations: EvalRun[] }>(`/api/papers/${paperId}/evaluations`).then((r) => r.evaluations),
  runEvaluation: (paperId: string, dataset?: string, k?: number) => {
    const q = new URLSearchParams();
    if (dataset) q.set("dataset", dataset);
    if (k) q.set("k", String(k));
    const qs = q.toString();
    return post<{ job_id: string; dataset: string }>(
      `/api/papers/${paperId}/evaluate${qs ? `?${qs}` : ""}`
    );
  },
  listExperiments: (paperId: string) =>
    get<{ experiments: Experiment[] }>(`/api/papers/${paperId}/experiments`).then(
      (r) => r.experiments
    ),
  runExperiments: (paperId: string) => post<{ job_id: string; dataset: string }>(
    `/api/papers/${paperId}/experiments`
  ),

  // jobs
  getJob: (jobId: string) => get<Job>(`/api/jobs/${jobId}`),
  listJobs: (paperId?: string) =>
    get<{ jobs: Job[] }>(
      paperId ? `/api/papers/${paperId}/jobs` : "/api/jobs"
    ).then((r) => r.jobs),
};
