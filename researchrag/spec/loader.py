"""Load an ImplementationSpec from the database as a typed model.

The API stores specs as rows; code generation needs the ``ImplementationSpec``
model. This is the single reconstruction point (shared by API and CLI).
"""

from __future__ import annotations

from researchrag.models.retrieval import (
    ImplementationSpec,
    RequirementStatus,
    RequirementType,
    SourceKind,
    SpecRequirement,
)


def spec_from_db(ctx, spec_id: str) -> ImplementationSpec | None:
    row = ctx.db.get_spec(spec_id)
    if row is None:
        return None
    requirements = []
    for r in row.get("requirements", []):
        try:
            rtype = RequirementType(r["type"])
        except ValueError:
            rtype = RequirementType.ARCHITECTURE
        try:
            source = SourceKind(r["source"])
        except ValueError:
            source = SourceKind.INFERRED
        try:
            status = RequirementStatus(r["status"])
        except (ValueError, KeyError, TypeError):
            status = RequirementStatus.OPEN
        requirements.append(
            SpecRequirement(
                id=r["req_id"],
                requirement=r["requirement"],
                type=rtype,
                source=source,
                confidence=r.get("confidence", 0.5),
                implementation_implication=r.get("implication", ""),
                status=status,
                source_section=r.get("source_section"),
                source_page=r.get("source_page"),
                source_chunk_id=r.get("source_chunk_id"),
                source_quote=r.get("source_quote"),
                source_ref=r.get("source_ref"),
                notes=r.get("notes", "") or "",
            )
        )
    return ImplementationSpec(
        id=row["id"],
        paper_id=row["paper_id"],
        title=row.get("title") or "",
        summary=row.get("summary") or "",
        requirements=requirements,
        task_decomposition=row.get("task_decomposition", []),
        dependencies=row.get("dependencies", []),
        open_questions=row.get("open_questions", []),
        generated_by=row.get("generated_by", ""),
        created_at=row.get("created_at", ""),
        version=row.get("version", 1),
    )
