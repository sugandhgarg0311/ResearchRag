"""SQLite persistence for application state.

Why SQLite (and not Postgres/pgvector)
--------------------------------------
* Runs locally with zero services — a single file next to the Qdrant data.
* This application's relational data is small and read-heavy (papers,
  blocks, specs, runs). Postgres adds an operational dependency without
  a real benefit at this scale. If you later need multi-user concurrency,
  the repository methods here are the seam to swap.

The vector index lives in Qdrant (embedded local mode). The two stores are
intentionally separate: Qdrant for similarity search + chunk payloads,
SQLite for structured application state. See PROJECT_ARCHITECTURE.md.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    title TEXT,
    authors_json TEXT NOT NULL DEFAULT '[]',
    abstract TEXT,
    page_count INTEGER NOT NULL DEFAULT 0,
    file_sha256 TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    source_path TEXT,
    parser TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    paper_id TEXT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    progress REAL NOT NULL DEFAULT 0.0,
    detail TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_paper ON jobs(paper_id);

CREATE TABLE IF NOT EXISTS blocks (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    page INTEGER NOT NULL,
    page_end INTEGER,
    "order" INTEGER NOT NULL,
    block_type TEXT NOT NULL,
    text TEXT NOT NULL,
    section_path_json TEXT NOT NULL DEFAULT '[]',
    section_id INTEGER,
    heading_level INTEGER,
    caption TEXT,
    image_path TEXT,
    bbox_json TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_blocks_paper ON blocks(paper_id);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    parent_id TEXT,
    section_path_json TEXT NOT NULL DEFAULT '[]',
    section_id INTEGER,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    block_ids_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);

CREATE TABLE IF NOT EXISTS specs (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    title TEXT,
    summary TEXT,
    task_decomposition_json TEXT NOT NULL DEFAULT '[]',
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    open_questions_json TEXT NOT NULL DEFAULT '[]',
    generated_by TEXT,
    created_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_specs_paper ON specs(paper_id);

CREATE TABLE IF NOT EXISTS spec_requirements (
    id TEXT PRIMARY KEY,
    spec_id TEXT NOT NULL,
    req_id TEXT NOT NULL,
    requirement TEXT NOT NULL,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    implication TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    source_section TEXT,
    source_page INTEGER,
    source_chunk_id TEXT,
    source_quote TEXT,
    source_ref TEXT,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_req_spec ON spec_requirements(spec_id);

CREATE TABLE IF NOT EXISTS implementations (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    spec_id TEXT,
    status TEXT NOT NULL DEFAULT 'generated',
    file_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_impl_paper ON implementations(paper_id);

CREATE TABLE IF NOT EXISTS implementation_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    implementation_id TEXT NOT NULL,
    path TEXT NOT NULL,
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'source'
);
CREATE INDEX IF NOT EXISTS idx_impl_files ON implementation_files(implementation_id);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    implementation_id TEXT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    command_json TEXT NOT NULL DEFAULT '[]',
    log_path TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT,
    finished_at TEXT,
    duration_ms INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_paper ON runs(paper_id);

CREATE TABLE IF NOT EXISTS reproduction_records (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    paper_value REAL,
    our_value REAL,
    difference REAL,
    dataset TEXT,
    seed TEXT,
    hardware TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_repro_paper ON reproduction_records(paper_id);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    paper_id TEXT,
    kind TEXT NOT NULL,
    config_json TEXT NOT NULL,
    results_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    duration_ms INTEGER
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    dataset TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    per_question_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_paper ON eval_runs(paper_id);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Database:
    """Thin, explicit wrapper over sqlite3 (WAL mode, thread-safe via a
    per-thread connection)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # generic helpers
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(sql, params).fetchall()

    def kv_get(self, key: str) -> str | None:
        rows = self.query("SELECT value FROM kv WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def kv_set(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------
    # papers
    def upsert_paper(self, **fields: Any) -> None:
        existing = self.query("SELECT id FROM papers WHERE id = ?", (fields["id"],))
        if existing:
            sets = ", ".join(f"{k} = ?" for k in fields if k != "id")
            params = tuple(fields[k] for k in fields if k != "id") + (fields["id"],)
            self.execute(f"UPDATE papers SET {sets} WHERE id = ?", params)
        else:
            cols = ", ".join(fields)
            ph = ", ".join("?" for _ in fields)
            self.execute(
                f"INSERT INTO papers ({cols}) VALUES ({ph})", tuple(fields.values())
            )

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM papers WHERE id = ?", (paper_id,))
        return _paper_row(rows[0]) if rows else None

    def list_papers(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM papers ORDER BY created_at DESC")
        return [_paper_row(r) for r in rows]

    def delete_paper(self, paper_id: str) -> None:
        with self._conn() as conn:
            for table in (
                "blocks", "chunks", "spec_requirements", "runs",
                "reproduction_records", "implementation_files",
                "eval_runs",
            ):
                if table == "spec_requirements":
                    conn.execute(
                        "DELETE FROM spec_requirements WHERE spec_id IN "
                        "(SELECT id FROM specs WHERE paper_id = ?)",
                        (paper_id,),
                    )
                elif table in ("implementation_files",):
                    conn.execute(
                        "DELETE FROM implementation_files WHERE implementation_id IN "
                        "(SELECT id FROM implementations WHERE paper_id = ?)",
                        (paper_id,),
                    )
                else:
                    conn.execute(f"DELETE FROM {table} WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM specs WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM implementations WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM jobs WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))

    # ------------------------------------------------------------------
    # blocks
    def insert_blocks(self, paper_id: str, blocks) -> int:
        rows = [
            (
                b.id, paper_id, b.page, b.page_end, b.order, b.block_type.value,
                b.text, json.dumps(b.section_path), b.section_id,
                b.heading_level, b.caption, b.image_path,
                json.dumps(b.bbox) if b.bbox else None, json.dumps(b.metadata),
            )
            for b in blocks
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO blocks "
                "(id, paper_id, page, page_end, \"order\", block_type, text, "
                "section_path_json, section_id, heading_level, caption, image_path, "
                "bbox_json, metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def get_blocks(self, paper_id: str) -> list[sqlite3.Row]:
        return self.query(
            'SELECT * FROM blocks WHERE paper_id = ? ORDER BY "order"', (paper_id,)
        )

    # ------------------------------------------------------------------
    # chunks
    def insert_chunks(self, paper_id: str, chunks) -> int:
        rows = [
            (
                c.id, paper_id, c.kind, c.chunk_type, c.text, c.token_count,
                c.parent_id, json.dumps(c.section_path), c.section_id,
                c.page_start, c.page_end, json.dumps(c.block_ids),
                json.dumps(c.metadata),
            )
            for c in chunks
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(id, paper_id, kind, chunk_type, text, token_count, parent_id, "
                "section_path_json, section_id, page_start, page_end, block_ids_json, "
                "metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def get_chunks(self, paper_id: str, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            rows = self.query(
                "SELECT * FROM chunks WHERE paper_id = ? AND kind = ? "
                "ORDER BY page_start, rowid",
                (paper_id, kind),
            )
        else:
            rows = self.query(
                "SELECT * FROM chunks WHERE paper_id = ? ORDER BY page_start, rowid",
                (paper_id,),
            )
        return [_chunk_row(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        return _chunk_row(rows[0]) if rows else None

    # ------------------------------------------------------------------
    # jobs
    def create_job(self, paper_id: str | None, kind: str) -> str:
        job_id = new_id("job")
        self.execute(
            "INSERT INTO jobs (id, paper_id, kind, status, created_at) "
            "VALUES (?, ?, ?, 'queued', ?)",
            (job_id, paper_id, kind, now_iso()),
        )
        return job_id

    def update_job(
        self,
        job_id: str,
        status: str | None = None,
        progress: float | None = None,
        detail: str | None = None,
        error: str | None = None,
    ) -> None:
        sets, params = [], []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
            if status in ("done", "failed"):
                sets.append("finished_at = ?")
                params.append(now_iso())
        if progress is not None:
            sets.append("progress = ?")
            params.append(progress)
        if detail is not None:
            sets.append("detail = ?")
            params.append(detail)
        if error is not None:
            sets.append("error = ?")
            params.append(error)
        if sets:
            params.append(job_id)
            self.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return dict(rows[0]) if rows else None

    def latest_job(self, paper_id: str, kind: str) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT * FROM jobs WHERE paper_id = ? AND kind = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (paper_id, kind),
        )
        return dict(rows[0]) if rows else None

    def list_jobs(self, paper_id: str | None = None) -> list[dict[str, Any]]:
        if paper_id:
            rows = self.query(
                "SELECT * FROM jobs WHERE paper_id = ? ORDER BY created_at DESC LIMIT 50",
                (paper_id,),
            )
        else:
            rows = self.query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50")
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # specs / requirements
    def insert_spec(self, spec) -> None:
        self.execute(
            "INSERT OR REPLACE INTO specs "
            "(id, paper_id, title, summary, task_decomposition_json, "
            "dependencies_json, open_questions_json, generated_by, created_at, version) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                spec.id, spec.paper_id, spec.title, spec.summary,
                json.dumps(spec.task_decomposition), json.dumps(spec.dependencies),
                json.dumps(spec.open_questions), spec.generated_by,
                spec.created_at, spec.version,
            ),
        )
        with self._conn() as conn:
            conn.execute("DELETE FROM spec_requirements WHERE spec_id = ?", (spec.id,))
            conn.executemany(
                "INSERT INTO spec_requirements "
                "(id, spec_id, req_id, requirement, type, source, confidence, "
                "implication, status, source_section, source_page, source_chunk_id, "
                "source_quote, source_ref, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        new_id("req"), spec.id, r.id, r.requirement, r.type.value,
                        r.source.value, r.confidence, r.implementation_implication,
                        r.status.value, r.source_section, r.source_page,
                        r.source_chunk_id, r.source_quote, r.source_ref, r.notes,
                    )
                    for r in spec.requirements
                ],
            )

    def get_spec(self, spec_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM specs WHERE id = ?", (spec_id,))
        if not rows:
            return None
        spec = dict(rows[0])
        spec["task_decomposition"] = json.loads(spec.pop("task_decomposition_json"))
        spec["dependencies"] = json.loads(spec.pop("dependencies_json"))
        spec["open_questions"] = json.loads(spec.pop("open_questions_json"))
        spec["requirements"] = [
            dict(r)
            for r in self.query(
                "SELECT * FROM spec_requirements WHERE spec_id = ? ORDER BY req_id",
                (spec_id,),
            )
        ]
        return spec

    def list_specs(self, paper_id: str) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT id, paper_id, title, generated_by, created_at, version "
            "FROM specs WHERE paper_id = ? ORDER BY created_at DESC",
            (paper_id,),
        )
        return [dict(r) for r in rows]

    def update_requirement_status(
        self, req_id: str, status: str, notes: str | None = None
    ) -> None:
        if notes is not None:
            self.execute(
                "UPDATE spec_requirements SET status = ?, notes = ? WHERE req_id = ?",
                (status, notes, req_id),
            )
        else:
            self.execute(
                "UPDATE spec_requirements SET status = ? WHERE req_id = ?",
                (status, req_id),
            )

    # ------------------------------------------------------------------
    # implementations / runs / reproduction
    def insert_implementation(
        self, impl_id: str, paper_id: str, spec_id: str | None, files: list[dict]
    ) -> None:
        self.execute(
            "INSERT OR REPLACE INTO implementations "
            "(id, paper_id, spec_id, status, file_count, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (impl_id, paper_id, spec_id, "generated", len(files), now_iso()),
        )
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO implementation_files (implementation_id, path, content, kind) "
                "VALUES (?,?,?,?)",
                [
                    (impl_id, f["path"], f["content"], f.get("kind", "source"))
                    for f in files
                ],
            )

    def get_implementation(self, impl_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM implementations WHERE id = ?", (impl_id,))
        if not rows:
            return None
        impl = dict(rows[0])
        impl["files"] = [
            dict(r)
            for r in self.query(
                "SELECT path, content, kind FROM implementation_files "
                "WHERE implementation_id = ? ORDER BY path",
                (impl_id,),
            )
        ]
        return impl

    def list_implementations(self, paper_id: str) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT id, paper_id, spec_id, status, file_count, created_at "
            "FROM implementations WHERE paper_id = ? ORDER BY created_at DESC",
            (paper_id,),
        )
        return [dict(r) for r in rows]

    def insert_run(self, run: dict[str, Any]) -> None:
        self.execute(
            "INSERT OR REPLACE INTO runs "
            "(id, paper_id, implementation_id, kind, status, command_json, log_path, "
            "metrics_json, started_at, finished_at, duration_ms, error) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run["id"], run["paper_id"], run.get("implementation_id"), run["kind"],
                run["status"], json.dumps(run.get("command", [])), run.get("log_path"),
                json.dumps(run.get("metrics", {})), run.get("started_at"),
                run.get("finished_at"), run.get("duration_ms"), run.get("error"),
            ),
        )

    def list_runs(self, paper_id: str) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM runs WHERE paper_id = ? ORDER BY started_at DESC",
            (paper_id,),
        )
        out = []
        for r in rows:
            d = dict(r)
            d["command"] = json.loads(d.pop("command_json"))
            d["metrics"] = json.loads(d.pop("metrics_json"))
            out.append(d)
        return out

    def insert_reproduction_record(self, record: dict[str, Any]) -> str:
        rec_id = new_id("repro")
        self.execute(
            "INSERT INTO reproduction_records "
            "(id, paper_id, metric, paper_value, our_value, difference, dataset, "
            "seed, hardware, config_json, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec_id, record["paper_id"], record["metric"], record.get("paper_value"),
                record.get("our_value"), record.get("difference"), record.get("dataset"),
                record.get("seed"), record.get("hardware"),
                json.dumps(record.get("config", {})), record.get("notes"), now_iso(),
            ),
        )
        return rec_id

    def list_reproduction_records(self, paper_id: str) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM reproduction_records WHERE paper_id = ? ORDER BY created_at",
            (paper_id,),
        )
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d.pop("config_json"))
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # experiments / evals
    def insert_experiment(self, exp: dict[str, Any]) -> None:
        self.execute(
            "INSERT OR REPLACE INTO experiments "
            "(id, name, paper_id, kind, config_json, results_json, created_at, duration_ms) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                exp["id"], exp["name"], exp.get("paper_id"), exp["kind"],
                json.dumps(exp["config"]), json.dumps(exp["results"]),
                exp.get("created_at", now_iso()), exp.get("duration_ms"),
            ),
        )

    def list_experiments(self, paper_id: str | None = None) -> list[dict[str, Any]]:
        if paper_id:
            rows = self.query(
                "SELECT * FROM experiments WHERE paper_id = ? ORDER BY created_at DESC",
                (paper_id,),
            )
        else:
            rows = self.query("SELECT * FROM experiments ORDER BY created_at DESC")
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d.pop("config_json"))
            d["results"] = json.loads(d.pop("results_json"))
            out.append(d)
        return out

    def insert_eval_run(self, ev: dict[str, Any]) -> None:
        self.execute(
            "INSERT OR REPLACE INTO eval_runs "
            "(id, paper_id, dataset, config_json, metrics_json, per_question_json, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                ev["id"], ev["paper_id"], ev["dataset"], json.dumps(ev["config"]),
                json.dumps(ev["metrics"]), json.dumps(ev.get("per_question", [])),
                ev.get("created_at", now_iso()),
            ),
        )

    def list_eval_runs(self, paper_id: str | None = None) -> list[dict[str, Any]]:
        if paper_id:
            rows = self.query(
                "SELECT * FROM eval_runs WHERE paper_id = ? ORDER BY created_at DESC",
                (paper_id,),
            )
        else:
            rows = self.query("SELECT * FROM eval_runs ORDER BY created_at DESC")
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d.pop("config_json"))
            d["metrics"] = json.loads(d.pop("metrics_json"))
            d["per_question"] = json.loads(d.pop("per_question_json"))
            out.append(d)
        return out


def _paper_row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["authors"] = json.loads(d.pop("authors_json", "[]"))
    return d


def _chunk_row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["section_path"] = json.loads(d.pop("section_path_json"))
    d["block_ids"] = json.loads(d.pop("block_ids_json"))
    d["metadata"] = json.loads(d.pop("metadata_json"))
    return d
