"""Windows portability for generated-code execution.

Guards the fixes that make `implement`/`run` work without a POSIX shell
and without a UTF-8 locale:
* scaffold templates read config.yaml as UTF-8 and run_smoke.sh uses a
  valid BASH_SOURCE reference plus an explicit PYTHON_BIN;
* the sandbox scrubs to UTF-8 and decodes child output as UTF-8;
* validation falls back to direct python steps when bash is missing.
"""

from __future__ import annotations

import sys

from researchrag.codegen import scaffold
from researchrag.execution import runner


def test_smoke_template_reads_config_as_utf8():
    src = scaffold._smoke_test("demo-paper")
    assert src.count('open("config.yaml", encoding="utf-8")') == 2
    assert 'open("config.yaml")' not in src.replace('open("config.yaml", encoding="utf-8")', "")


def test_run_smoke_template_valid_bash_and_python_bin():
    src = scaffold._run_smoke("demo-paper")
    assert '"${BASH_SOURCE[0]}"' in src
    assert '"{BASH_SOURCE[0]}"' not in src
    assert '"${PYTHON_BIN:-python}"' in src


def test_scrubbed_env_forces_utf8():
    env = runner._scrubbed_env()
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert not any("API_KEY" in k for k in env)


def test_run_command_decodes_utf8_output(tmp_path):
    res = runner.run_command(
        [sys.executable, "-c", "print('Snowman \\u2603 and theta \\u03b8')"],
        cwd=tmp_path,
        timeout_s=30,
    )
    assert res.returncode == 0
    assert "Snowman ☃ and theta θ" in res.stdout


def test_validator_passes_when_unittest_writes_ok_to_stderr(tmp_path, monkeypatch):
    """unittest writes its 'OK' summary to stderr, not stdout.

    Regression: _run_checks judged pass/fail on stdout only, so a passing
    suite (returncode 0, 'OK' on stderr, empty stdout) was reported failed.
    """
    import researchrag.codegen.validator as validator_mod

    monkeypatch.setattr("researchrag.execution.runner.shell_available", lambda: False)
    tests = tmp_path / "tests"
    tests.mkdir()
    # Mimic `python -m unittest`: summary on stderr, stdout empty.
    (tests / "test_smoke.py").write_text(
        "import sys\nsys.stderr.write('Ran 3 tests in 0.036s\\n\\nOK\\n')\n",
        encoding="utf-8",
    )
    (tmp_path / "run_smoke.sh").write_text("#!/usr/bin/env bash\necho SMOKE_OK\n", encoding="utf-8")

    stages = validator_mod._run_checks(tmp_path, sys.executable, timeout_s=60)
    by_name = {s.name: s for s in stages}
    assert by_name["tests"].status == "passed"


def test_validator_falls_back_without_bash(tmp_path, monkeypatch):
    import researchrag.codegen.validator as validator_mod

    monkeypatch.setattr("researchrag.execution.runner.shell_available", lambda: False)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_smoke.py").write_text("print('OK')\n", encoding="utf-8")
    (tmp_path / "run_smoke.sh").write_text("#!/usr/bin/env bash\necho SMOKE_OK\n", encoding="utf-8")

    stages = validator_mod._run_checks(tmp_path, sys.executable, timeout_s=60)
    by_name = {s.name: s for s in stages}
    assert by_name["tests"].status == "passed"
    assert by_name["smoke"].status == "passed"
    assert "no bash" in by_name["smoke"].detail
