"""Generated-code validation.

Stages (each reported independently):

1. **syntax** — every .py file must parse (ast).
2. **imports** — intra-project imports must resolve to real files.
3. **tests** — the generated smoke test suite must run and pass.
4. **smoke run** — the entry script runs within timeout/memory limits.

A validation result is never a binary "success": each stage has its own
status so the UI can show exactly what was checked. "Smoke passed" is
reported as smoke-passed — never as "the paper is reproduced".
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageResult:
    name: str
    status: str  # passed | failed | skipped
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class ValidationResult:
    ok: bool
    stages: list[StageResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def stage(self, name: str) -> StageResult | None:
        return next((s for s in self.stages if s.name == name), None)


def validate_project(root: Path, python: str | None = None, timeout_s: int = 120) -> ValidationResult:
    root = Path(root)
    result = ValidationResult(ok=True)
    errors: list[str] = []

    # ---------------- syntax -------------------------------------------
    py_files = sorted(root.rglob("*.py"))
    syntax_errors = []
    for f in py_files:
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            syntax_errors.append(f"{f.relative_to(root)}: {e.msg} (line {e.lineno})")
    if syntax_errors:
        result.ok = False
        errors.extend(syntax_errors)
    result.stages.append(
        StageResult(
            "syntax",
            "failed" if syntax_errors else "passed",
            f"{len(py_files)} files checked" + (
                f"; {len(syntax_errors)} errors" if syntax_errors else ""
            ),
        )
    )
    if syntax_errors:
        result.errors = errors
        return result

    # ---------------- imports -------------------------------------------
    pkg_dirs = {d.name.replace("-", "_"): d for d in (root / "src").iterdir() if d.is_dir()}
    import_errors = []
    for f in py_files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in pkg_dirs:
                    target = root / "src" / top
                    if not target.exists():
                        import_errors.append(f"{f.relative_to(root)}: missing package '{top}'")
    if import_errors:
        result.ok = False
        errors.extend(import_errors)
    result.stages.append(
        StageResult(
            "imports",
            "failed" if import_errors else "passed",
            "intra-project imports resolve" if not import_errors else "; ".join(import_errors[:3]),
        )
    )

    if result.ok:
        result.stages.extend(_run_checks(root, python or sys.executable, timeout_s))
        if not all(s.status == "passed" for s in result.stages if s.name in ("tests", "smoke")):
            result.ok = any(s.status == "failed" for s in result.stages)
            # tests failing means not ok
            if result.stage("tests") and result.stage("tests").status == "failed":
                result.ok = False
            for s in result.stages:
                if s.status == "failed":
                    errors.append(f"[{s.name}] {s.detail}")
    result.errors = errors
    return result


def _run_checks(root: Path, python: str, timeout_s: int) -> list[StageResult]:
    # tests
    import time

    from researchrag.execution.runner import run_command, shell_available

    t0 = time.perf_counter()
    test_file = root / "tests" / "test_smoke.py"
    if test_file.exists():
        run = run_command(
            [python, str(test_file)],
            cwd=root,
            timeout_s=min(timeout_s, 120),
        )
        ms = (time.perf_counter() - t0) * 1000
        # NOTE: unittest writes its result summary ("OK"/"FAILED") to stderr,
        # not stdout — so judge pass/fail on the combined streams, not stdout
        # alone. (A passing suite with empty stdout previously reported
        # "failed" with detail "exit=0: ... OK".)
        combined = f"{run.stdout or ''}\n{run.stderr or ''}"
        ok = run.returncode == 0 and "OK" in combined
        detail = (
            "smoke tests passed"
            if ok
            else f"exit={run.returncode}: {(run.stderr or run.stdout).strip()[-300:]}"
        )
        test_stage = StageResult("tests", "passed" if ok else "failed", detail, ms)
    else:
        test_stage = StageResult("tests", "skipped", "no test file generated")

    # smoke run script (compileall + tests again as a script)
    t0 = time.perf_counter()
    smoke = root / "run_smoke.sh"
    if smoke.exists():
        os.chmod(smoke, 0o755)
        env = {"PYTHON_BIN": python}
        if shell_available():
            run = run_command(
                ["bash", str(smoke)],
                cwd=root,
                timeout_s=min(timeout_s, 180),
                extra_env=env,
            )
            via = "run_smoke.sh"
        else:
            # No POSIX shell (plain Windows): perform exactly the steps the
            # script performs — syntax check, then the smoke tests.
            run = run_command(
                [python, "-m", "compileall", "-q", "."],
                cwd=root,
                timeout_s=min(timeout_s, 180),
                extra_env=env,
            )
            via = "python fallback (no bash)"
            if run.returncode == 0:
                run = run_command(
                    [python, str(test_file)] if test_file.exists() else [python, "-c", "pass"],
                    cwd=root,
                    timeout_s=min(timeout_s, 180),
                    extra_env=env,
                )
        ms = (time.perf_counter() - t0) * 1000
        # Same combined-stream rule as the tests stage above: unittest and
        # the smoke script report via either stream.
        combined = f"{run.stdout or ''}\n{run.stderr or ''}"
        ok = run.returncode == 0 and ("SMOKE_OK" in combined or "OK" in combined)
        detail = f"{via} completed (SMOKE_OK)" if ok else (
            f"exit={run.returncode}: {(run.stderr or run.stdout).strip()[-300:]}"
        )
        smoke_stage = StageResult("smoke", "passed" if ok else "failed", detail, ms)
    else:
        smoke_stage = StageResult("smoke", "skipped", "no run_smoke.sh")
    return [test_stage, smoke_stage]
