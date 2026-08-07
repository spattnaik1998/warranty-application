"""Rendering is where a careful measurement becomes a claim someone quotes.

These tests pin the two claims the report must never make: a dollar figure with
no declared volume behind it, and a measured-looking $0.00 when cost could not be
measured at all.
"""

from __future__ import annotations

from warrant.analysis.report import AuditReport, NodeFinding, Recommendation
from warrant.report import render_html
from warrant.schemas.belief import AdmissibilityClass


def _finding(node_id: str, **kwargs) -> NodeFinding:
    base = dict(
        node_id=node_id,
        admissibility=AdmissibilityClass.REORGANIZER,
        recommendation=Recommendation.COLLAPSE,
        confidence=0.75,
        ablation_tested=True,
        ablation_value=0.0,
        ablation_runs=12,
        mean_novelty=0.05,
        model="gpt-4o",
        dollars_per_1k_runs=1.42,
        projected_savings_per_1k_runs=1.42,
        reason="reorganizer with zero delegation value.",
    )
    base.update(kwargs)
    return NodeFinding(**base)


def _report(**kwargs) -> AuditReport:
    base = dict(
        graph_name="demo",
        n_runs=12,
        findings=[_finding("reviewer")],
        total_dollars_per_1k_runs=5.42,
        projected_savings_per_1k_runs=1.42,
    )
    base.update(kwargs)
    return AuditReport(**base)


def test_measured_unit_is_per_1k_runs_without_a_volume() -> None:
    report = _report()
    cli, html = report.to_cli(), render_html(report)
    assert "$/1k runs" in cli
    assert "per 1,000 runs" in report.savings_sentence()
    assert "/mo" not in report.savings_sentence()
    assert "$/1k runs" in html


def test_declared_volume_is_labelled_as_declared() -> None:
    report = _report(
        runs_per_month=30_000,
        findings=[_finding("reviewer", dollars_per_month=42.60,
                           projected_savings_per_month=42.60)],
        total_dollars_per_month=162.60,
        projected_savings_per_month=42.60,
    )
    sentence = report.savings_sentence()
    assert "$42.60/mo" in sentence
    assert "30,000 runs/month (declared)" in sentence
    assert "$/mo" in report.to_cli()
    assert "declared" in render_html(report)


def test_ablation_n_travels_with_the_value() -> None:
    """0.00 over 1 run and over 50 are different claims — both surfaces say which."""
    report = _report(findings=[_finding("reviewer", ablation_runs=1, confidence=0.2)])
    assert "(n=1)" in report.to_cli()
    assert "n=1" in render_html(report)


def test_unmeasurable_cost_is_absent_not_zero() -> None:
    """A static scan must not render $0.00 — that reads as a measured zero."""
    report = _report(
        economics_available=False,
        findings=[_finding("reviewer", recommendation=Recommendation.REVIEW,
                           ablation_tested=False, ablation_value=None,
                           dollars_per_1k_runs=0.0, projected_savings_per_1k_runs=0.0)],
        total_dollars_per_1k_runs=0.0,
        projected_savings_per_1k_runs=0.0,
    )
    cli, html = report.to_cli(), render_html(report)
    assert "$" not in report.savings_sentence()
    assert "$0.00" not in cli and "$0.00" not in html
    assert "Run the graph to prove and price them" in report.savings_sentence()
    assert "<svg" not in html                 # no empty bar chart implying zero
    # The cost columns are dropped, not blanked — a column of dashes still reads
    # as "we tried to price this and got nothing".
    assert "$/1k runs" not in cli and "$/1k runs" not in html
    assert "<th>Model</th>" not in html


def test_static_headline_counts_candidates_not_collapses() -> None:
    """A static scan can never emit COLLAPSE, so counting those always said 0."""
    report = _report(
        economics_available=False,
        findings=[
            _finding("reviewer", recommendation=Recommendation.REVIEW,
                     ablation_tested=False, ablation_value=None),
            _finding("analyze", recommendation=Recommendation.REVIEW,
                     ablation_tested=False, ablation_value=None),
            _finding("retriever", admissibility=AdmissibilityClass.INJECTOR,
                     recommendation=Recommendation.KEEP,
                     ablation_tested=False, ablation_value=None),
        ],
    )
    assert report.collapsible() == []                     # nothing is proven
    assert len(report.candidates()) == 2                  # but two are worth a look
    assert report.savings_sentence().startswith("2 reorganizer candidate(s)")


def test_static_headline_when_every_node_injects() -> None:
    report = _report(
        economics_available=False,
        findings=[_finding("retriever", admissibility=AdmissibilityClass.INJECTOR,
                           recommendation=Recommendation.KEEP,
                           ablation_tested=False, ablation_value=None)],
    )
    assert "No reorganizer candidates found" in report.savings_sentence()


def test_absent_distortion_is_null_in_json_not_zero() -> None:
    """`0.0` in JSON reads as "no distortion"; the honest value is null."""
    payload = _report().to_dict()
    assert payload["distortion_available"] is False
    assert payload["mean_chain_loss_bits"] is None


def test_html_is_self_contained_and_escapes_user_text() -> None:
    report = _report(findings=[_finding("<script>alert(1)</script>")])
    html = render_html(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "http://" not in html and "https://cdn" not in html   # no external assets
