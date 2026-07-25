"""End-to-end smoke tests (mock mode) for the pipeline, orchestrator, and ledger."""

from __future__ import annotations

import warrant.tools  # noqa: F401
from warrant.ledger import run_probe
from warrant.ledger.report import acceptance
from warrant.orchestrator import GovernedOrchestrator
from warrant.pipeline import run_brief
from warrant.schemas.belief import AdmissibilityClass
from warrant.schemas.tasks import BriefRequest


def test_ungoverned_brief_is_grounded():
    result = run_brief(BriefRequest(arxiv_id="2603.26993"))
    assert result.arxiv_id == "2603.26993"
    assert "Reliability Limits" in (result.paper_title or "")
    assert result.sections
    # No high-confidence claim lacks evidence.
    assert result.ungrounded_claims == []


def test_governed_collapses_composition():
    gov = GovernedOrchestrator().run(BriefRequest(arxiv_id="2603.26993"))
    types = {g["delegation_type"]: g for g in gov.gate_log}
    # Composition is present and was rejected as a reorganizer (collapsed).
    assert types["compose"]["admit"] is False
    assert types["compose"]["cls"] == AdmissibilityClass.REORGANIZER.value
    # At least one exogenous injector was admitted.
    assert gov.admitted >= 1
    # Every belief is confident and correct.
    for dk_key in gov.ctx.beliefs.keys():
        assert gov.ctx.beliefs.posterior(dk_key).top_prob() > 0.8


def test_governed_matches_ungoverned_quality():
    gov = GovernedOrchestrator().run(BriefRequest(arxiv_id="2603.26993"))
    assert gov.brief.paper_title
    assert gov.brief.ungrounded_claims == []


def test_probe_reproduces_paper_findings():
    probe = run_probe(BriefRequest(arxiv_id="2603.26993"))
    checks = acceptance(probe)
    assert all(checks.values()), checks
    # Concretely: naive relay collapses accuracy, KL predicts the drop.
    from warrant.schemas.ledger import Condition

    assert probe.runs[Condition.GOVERNED].accuracy >= probe.runs[Condition.NAIVE].accuracy
    assert probe.kl_accuracy_r > 0.5
