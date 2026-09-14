"""Code generation service: spec → project files → validation.

* The deterministic scaffold is ALWAYS produced (it is the runnable base).
* When an LLM is configured, individual source files can be (re)generated
  with full spec + evidence context; failures fall back to the scaffold
  file, so generation is never all-or-nothing.
* Every generated project is validated (syntax/imports/tests/smoke) before
  it is reported as ready.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from researchrag.codegen.scaffold import GeneratedFile, generate_scaffold
from researchrag.codegen.validator import ValidationResult, validate_project
from researchrag.config import Settings
from researchrag.llm.base import BaseLLM, LLMMessage
from researchrag.models.retrieval import ImplementationSpec
from researchrag.storage.database import Database, new_id

log = logging.getLogger(__name__)

FILE_PROMPT = """You are implementing one file of a project that reproduces a
research paper's method. The project was scaffolded from an implementation
spec; every requirement carries an id (REQ-xxx) and provenance.

Write ONLY the complete contents of the file `{path}`. Rules:
- Python 3.10+, standard library + the project's requirements.txt.
- Honor the tagged requirements (TODO(REQ-xxx) markers in the scaffold).
- Where the paper leaves a value unspecified, use a clearly commented
  default and mention it in the module docstring.
- No placeholder 'pass' bodies for required logic; implement it.
- Keep functions import-safe (no top-level training runs).
- End the file with a short docstring note of which REQ ids it covers.

Implementation spec (JSON):
{spec_json}

Relevant paper evidence:
{evidence}

Scaffold version of {path} (preserve its structure where it makes sense):
{scaffold}
"""


class CodeGenerator:
    def __init__(self, settings: Settings, db: Database, llm: BaseLLM | None = None):
        self.settings = settings
        self.db = db
        self.llm = llm

    # ------------------------------------------------------------------
    def generate(
        self,
        spec: ImplementationSpec,
        evidence_texts: list[tuple[str, str, list[str], int | None, str]] | None = None,
    ) -> tuple[str, list[GeneratedFile], ValidationResult]:
        files = generate_scaffold(spec)
        if self.llm is not None:
            files = self._llm_enhance(spec, files, evidence_texts or [])
        impl_id = new_id("impl")
        return impl_id, files, None  # validation done after writing

    def generate_and_validate(
        self,
        spec: ImplementationSpec,
        evidence_texts: list[tuple[str, str, list[str], int | None, str]] | None = None,
    ) -> tuple[str, ValidationResult, Path]:
        t0 = time.perf_counter()
        impl_id, files, _ = self.generate(spec, evidence_texts)
        project_dir = self._project_dir(impl_id)
        self._write_files(project_dir, files)
        validation = validate_project(project_dir, timeout_s=self.settings.execution_timeout_s)
        self.db.insert_implementation(
            impl_id,
            spec.paper_id,
            spec.id,
            [
                {"path": f.path, "content": f.content, "kind": f.kind}
                for f in files
            ],
        )
        log.info(
            "implementation %s: %d files, validation=%s (%.0fms)",
            impl_id,
            len(files),
            "ok" if validation.ok else "issues",
            (time.perf_counter() - t0) * 1000,
        )
        return impl_id, validation, project_dir

    # ------------------------------------------------------------------
    def _project_dir(self, impl_id: str) -> Path:
        return self.settings.artifacts_root / "implementations" / impl_id

    @staticmethod
    def _write_files(root: Path, files: list[GeneratedFile]) -> None:
        for f in files:
            target = root / f.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.content, encoding="utf-8")
            if f.kind == "script":
                target.chmod(0o755)

    # ------------------------------------------------------------------
    def _llm_enhance(
        self,
        spec: ImplementationSpec,
        files: list[GeneratedFile],
        evidence_texts: list[tuple[str, str, list[str], int | None, str]],
    ) -> list[GeneratedFile]:
        import json

        evidence = "\n\n".join(
            f"[{' › '.join(sp) or 'front'} p.{pg}] {tx[:600]}"
            for _cid, tx, sp, pg, _t in evidence_texts[:14]
        )
        spec_json = json.dumps(
            {
                "summary": spec.summary,
                "requirements": [
                    {
                        "id": r.id,
                        "type": r.type.value,
                        "source": r.source.value,
                        "requirement": r.requirement,
                        "implication": r.implementation_implication,
                    }
                    for r in spec.requirements
                ],
                "open_questions": spec.open_questions,
            },
            indent=1,
        )
        source_files = [f for f in files if f.path.startswith("src/") and f.path.endswith(".py") and f.path != "src/__init__.py" and "/__init__.py" not in f.path]
        out = []
        for f in files:
            if f in source_files:
                try:
                    resp = self.llm.chat(
                        [
                            LLMMessage(
                                role="system",
                                content=FILE_PROMPT.format(
                                    path=f.path,
                                    spec_json=spec_json,
                                    evidence=evidence or "(none)",
                                    scaffold=f.content,
                                ),
                            ),
                            LLMMessage(role="user", content=f"Return the file {f.path}."),
                        ],
                        temperature=0.15,
                        max_tokens=self.settings.llm_max_tokens,
                    )
                    content = _extract_code(resp.text, f.content)
                    out.append(GeneratedFile(f.path, content, kind=f.kind))
                    log.info("LLM-generated %s (%d chars)", f.path, len(content))
                except Exception as e:
                    log.warning("LLM file generation failed for %s (%s); scaffold kept", f.path, e)
                    out.append(f)
            else:
                out.append(f)
        return out


def _extract_code(text: str, fallback: str) -> str:
    import re

    text = text.strip()
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    if "def " in text or "import " in text or '"""' in text:
        return text if text.endswith("\n") else text + "\n"
    return fallback
