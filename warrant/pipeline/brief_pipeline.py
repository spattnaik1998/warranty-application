"""Ungoverned end-to-end brief pipeline (the first-priority happy path).

This runs the domain task directly, without the admissibility gate, and returns
a grounded :class:`BriefResult`. The governed orchestrator (slice 6) reuses the
same step primitives but decides delegations through the gate.
"""

from __future__ import annotations

from warrant.logging_setup import get_logger, log_event
from warrant.pipeline.steps import (
    PipelineContext,
    compose_sections,
    decision_keys,
    discover_target,
    elicit_key,
    fetch_paper,
    new_context,
    read_paper,
)
from warrant.schemas.belief import AdmissibilityClass
from warrant.schemas.tasks import BriefRequest, BriefResult, Section  # noqa: F401

log = get_logger("pipeline")


def _evidence_context(ctx: PipelineContext) -> str:
    """Assemble the full exogenous evidence available to the decision points."""
    parts = [
        ctx.artifacts.get("title", ""),
        ctx.artifacts.get("authors", ""),
        ctx.artifacts.get("abstract", ""),
        ctx.artifacts.get("paper_text", ""),
    ]
    return "\n".join(p for p in parts if p)


def ground_and_graycheck(ctx: PipelineContext, sections: list[Section]) -> tuple[list, list[str]]:
    """Grounding guard + gray-error check.

    * Grounding: a supported/quoted claim with no evidence_ref is a violation.
    * Gray error: a claim quoting a metric whose figure does not appear in the
      source text is a silent semantic failure.
    """
    ungrounded = []
    warnings: list[str] = []
    source = ctx.artifacts.get("paper_text", "").lower()
    for section in sections:
        for claim in section.claims:
            if not claim.grounded and claim.confidence >= 0.5:
                ungrounded.append(claim)
            # Gray-error: verify any quoted metric actually appears in source.
            for ref in claim.evidence_refs:
                if ref.locator == "metric" and ref.snippet:
                    if source and ref.snippet.lower() not in source:
                        warnings.append(
                            f"gray-error: quoted metric {ref.snippet!r} not found in source"
                        )
    if not source:
        warnings.append("source text unavailable — metric grounding could not be verified")
    return ungrounded, warnings


def run_brief_ctx(request: BriefRequest) -> tuple[BriefResult, PipelineContext]:
    """Run the pipeline and return both the result and the populated context."""
    ctx = new_context(request)
    arxiv_id = discover_target(ctx)
    fetch_paper(ctx, arxiv_id)
    read_paper(ctx, arxiv_id)

    dks = decision_keys(ctx)
    evidence = _evidence_context(ctx)
    for dk in dks:
        # In the ungoverned happy path, every decision sees the full evidence.
        elicit_key(ctx, dk, evidence,
                   delegation_type="resolve_" + dk.key.split("::")[0],
                   cls=AdmissibilityClass.INJECTOR)

    sections = compose_sections(ctx, dks)
    ungrounded, warnings = ground_and_graycheck(ctx, sections)
    from warrant.orchestrator.escalation import flag_for_review

    result = BriefResult(
        title=f"Technical briefing: {ctx.artifacts.get('title', 'Untitled')}",
        paper_title=ctx.artifacts.get("title"),
        arxiv_id=arxiv_id,
        sections=sections,
        ungrounded_claims=ungrounded,
        flagged_for_review=flag_for_review(ctx.beliefs, dks),
        warnings=warnings,
    )
    log_event(log, "brief complete", stage="finish", status="ok",
              arxiv_id=arxiv_id, duration_ms=0)
    return result, ctx


def run_brief(request: BriefRequest) -> BriefResult:
    """Public entry point: produce a technical briefing for a request."""
    result, _ = run_brief_ctx(request)
    return result
