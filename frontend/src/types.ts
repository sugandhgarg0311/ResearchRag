// TypeScript mirrors of the FastAPI response models.
// Field names must match researchrag/storage/database.py and the route handlers.

export interface Paper {
  id: string;
  title: string | null;
  filename: string;
  authors?: string[];
  page_count: number;
  file_sha256?: string | null;
  size_bytes?: number | null;
  parser?: string | null;
  status: string; // queued | ready | failed
  error?: string | null;
  created_at: string;
  jobs?: Job[];
}

export interface Job {
  id: string;
  paper_id: string | null;
  kind: string; // ingest | spec | implement | run | evaluate | experiment
  status: string; // queued | running | done | failed
  progress: number;
  detail: string | null;
  error: string | null;
  created_at: string;
  result?: unknown | null;
}

export interface StructureBlock {
  id: string;
  type: string;
  text: string;
  section: string[];
  section_id: number | null;
  heading_level: number | null;
  caption: string | null;
  has_image: boolean;
  bbox: number[] | null;
}

export interface StructurePage {
  number: number;
  blocks: StructureBlock[];
}

export interface StructureSection {
  id: number;
  level: number;
  path: string[];
  title: string;
  page: number;
  block_id: string;
}

export interface PaperStructure {
  paper_id: string;
  page_count: number;
  pages: StructurePage[];
  sections: StructureSection[];
}

export interface Chunk {
  id: string;
  paper_id: string;
  paper_title: string | null;
  kind: "child" | "parent";
  text: string;
  token_count: number;
  parent_id: string | null;
  section_path: string[];
  page_start: number;
  page_end: number;
  block_ids: string[];
  chunk_type: string;
  metadata: Record<string, unknown>;
}

// --- retrieval ----------------------------------------------------------------

export interface RetrievalConfig {
  dense_top_n?: number | null;
  sparse_top_n?: number | null;
  rrf_k?: number | null;
  rrf_dense_weight?: number | null;
  rrf_sparse_weight?: number | null;
  use_dense?: boolean;
  use_sparse?: boolean;
  use_rrf?: boolean;
  reranker_top_m?: number | null;
  final_top_k?: number | null;
  use_reranker?: boolean;
  use_parent_context?: boolean | null;
}

export interface RetrievedChunk {
  chunk: Chunk;
  dense_rank: number | null;
  dense_score: number | null;
  sparse_rank: number | null;
  sparse_score: number | null;
  rrf_score: number | null;
  rrf_rank: number | null;
  rerank_score: number | null;
  rerank_rank: number | null;
}

export interface StageTimings {
  dense_ms: number | null;
  sparse_ms: number | null;
  fusion_ms: number | null;
  rerank_ms: number | null;
  assembly_ms: number | null;
  total_ms: number;
}

export interface RetrievalResult {
  query: string;
  paper_id: string | null;
  config: RetrievalConfig;
  dense_candidates: number;
  sparse_candidates: number;
  fused_candidates: number;
  reranked_candidates: number;
  final_count: number;
  evidence: RetrievedChunk[];
  stages: StageTimings;
}

export interface Citation {
  chunk_id: string;
  paper_id: string;
  paper_title: string | null;
  section: string;
  page: number | null;
  page_range: string | null;
  quote: string | null;
  relevance: number | null;
  source: string;
}

export interface Answer {
  question: string;
  answer: string;
  mode: string; // llm | extractive
  model: string | null;
  citations: Citation[];
  evidence: Chunk[];
  sufficiency: string; // sufficient | partial | insufficient
  confidence: string; // explicit | strongly_inferred | weakly_inferred | external | unknown
  stages: StageTimings | null;
  tokens_used: number | null;
}

// --- specs -----------------------------------------------------------------------

export type ReqStatus = "open" | "accepted" | "rejected" | "implemented";

export interface SpecRequirement {
  id: string;
  spec_id: string;
  req_id: string;
  requirement: string;
  type: string; // model | dataset | hyperparameter | hardware | retrieval | ...
  source: string; // EXPLICIT | INFERRED | EXTERNAL | UNKNOWN
  confidence: number;
  implication: string;
  status: ReqStatus;
  source_section: string | null;
  source_page: number | null;
  source_chunk_id: string | null;
  source_quote: string | null;
  source_ref: string | null;
  notes: string;
}

export interface Spec {
  id: string;
  paper_id: string;
  title: string | null;
  summary: string | null;
  task_decomposition: unknown[];
  dependencies: unknown[];
  open_questions: string[];
  generated_by: string | null;
  created_at: string;
  version: number;
  requirements: SpecRequirement[];
}

export interface SpecSummary {
  id: string;
  paper_id: string;
  title: string | null;
  generated_by: string | null;
  created_at: string;
  version: number;
}

// --- implementations --------------------------------------------------------------

export interface ImplFileMeta {
  path: string;
  kind: string;
  size: number;
}

export interface ImplFile {
  path: string;
  content: string;
  kind: string;
}

export interface ValidationStage {
  name: string;
  status: string; // passed | failed | skipped
  detail: string;
}

export interface ImplementationSummary {
  id: string;
  paper_id: string;
  spec_id: string | null;
  status: string;
  file_count: number;
  created_at: string;
}

export interface Implementation {
  id: string;
  paper_id: string;
  spec_id: string | null;
  status: string;
  file_count: number;
  created_at: string;
  files: ImplFileMeta[];
  project_dir: string;
}

export interface RunRecord {
  id: string;
  paper_id: string;
  implementation_id: string;
  kind: string; // smoke | tests | train
  status: string; // ok | failed | timeout
  command: string[];
  log_path: string | null;
  metrics: Record<string, unknown>;
  started_at: string;
  finished_at: string;
  duration_ms: number;
  error: string | null;
}

export interface ReproductionRecord {
  id: string;
  paper_id: string;
  metric: string;
  paper_value: number | null;
  our_value: number | null;
  difference: number | null;
  dataset: string | null;
  seed: string | null;
  hardware: string | null;
  config: Record<string, unknown>;
  notes: string | null;
  created_at: string;
}

// --- evaluation ---------------------------------------------------------------------

export interface EvalRun {
  id: string;
  paper_id: string;
  dataset: string;
  k?: number;
  config: Record<string, unknown>;
  metrics: Record<string, number>;
  per_question: Array<Record<string, unknown>>;
  created_at: string;
}

export interface Experiment {
  id: string;
  name: string;
  paper_id: string | null;
  dataset: string | null;
  k: number | null;
  config: Record<string, unknown>;
  results: Record<string, unknown>;
  created_at: string;
}

// --- health ---------------------------------------------------------------------------

export interface Health {
  status: string;
  version: string;
  parser: string;
  embedding_model: string;
  embedding_fallback: boolean;
  reranker: string;
  llm: string;
  chunk_strategy: string;
  settings: Record<string, unknown>;
}
