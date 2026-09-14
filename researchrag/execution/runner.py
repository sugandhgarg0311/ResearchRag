"""Sandboxed command execution for generated code.

Security model (documented, pragmatic for a local research tool):

* Runs in a child process with a wall-clock timeout (SIGKILL).
* Working directory is confined to the project dir; the child receives a
  **scrubbed environment** — no parent API keys, no RESEARCHRAG_* secrets.
* On POSIX, RAM is capped via RLIMIT_AS in the child (preexec).
* Generated code is treated as UNTRUSTED. This is a resource/time sandbox,
  not a hard isolation boundary: for adversarial code run it inside a
  container. Network egress is not kernel-blocked; the scrubbed env
  removes credentials so project secrets cannot be exfiltrated.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    log_path: str | None = None


def _scrubbed_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        # UTF-8 mode: open() defaults to utf-8 in the child even on Windows
        # locales (cp1252). Without this, generated projects choke on the
        # UTF-8 quotes/ligatures that papers legitimately contain.
        "PYTHONUTF8": "1",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }


def shell_available() -> bool:
    """Is a POSIX shell present for run_smoke.sh? (Git Bash counts.)"""
    return shutil.which("bash") is not None


def _child_limits(memory_mb: int):
    """preexec_fn: apply resource limits in the child before exec."""
    if sys.platform in ("linux", "darwin"):

        def _apply():
            try:
                import resource

                limit = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except Exception:
                pass

        return _apply
    return None


def run_command(
    cmd: list[str],
    cwd: Path,
    timeout_s: int = 120,
    memory_mb: int = 1024,
    log_path: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> CommandResult:
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    timed_out = False
    env = _scrubbed_env()
    if extra_env:
        # Explicit per-call allowlist additions only (never parent secrets).
        env.update(extra_env)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            # Decode as UTF-8 explicitly: text=True would use the OS locale
            # (cp1252 on Windows) and crash on the UTF-8 the child emits.
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            preexec_fn=_child_limits(memory_mb) if os.name == "posix" else None,
        )
        out, err, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc = -1

    duration = (time.perf_counter() - t0) * 1000
    if timed_out:
        err = (err + f"\n[timed out after {timeout_s}s]").strip()

    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as f:
                f.write(f"$ {' '.join(cmd)}\n\n--- stdout ---\n{out}\n\n--- stderr ---\n{err}\n")
        except Exception:
            pass

    return CommandResult(
        returncode=rc,
        stdout=out,
        stderr=err,
        duration_ms=duration,
        timed_out=timed_out,
        log_path=str(log_path) if log_path else None,
    )
