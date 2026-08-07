"""Acceptance test: the audit distinguishes a redundant node from a load-bearing one.

Graph: retriever (injector) -> writer (reorganizer, load-bearing) -> reviewer
(reorganizer, redundant). Both writer and reviewer call no exogenous tool, so
the *structural* audit flags both as reorganizer candidates. Only *ablation*
tells them apart: disabling the writer changes the answer (keep); disabling the
reviewer never does (collapse). This nuance is the product's core value.
"""

from __future__ import annotations

from typing import TypedDict

import pytest

import warrant
from warrant.analysis.report import Recommendation

pytest.importorskip("langgraph.graph")


class _S(TypedDict):
    text: str


def build_graph(disabled: frozenset[str] = frozenset()):
    from langgraph.graph import END, StateGraph

    def retriever(s: _S) -> _S:
        return {"text": "papers: A, B, C"}

    def writer(s: _S) -> _S:  # load-bearing: produces the final briefing text
        return {"text": "BRIEF >> " + s["text"]}

    def reviewer(s: _S) -> _S:  # redundant: reads context, changes nothing material
        return {"text": s["text"]}

    fns = {"retriever": retriever, "writer": writer, "reviewer": reviewer}
    order = [n for n in ("retriever", "writer", "reviewer") if n not in disabled]

    g = StateGraph(_S)
    for name in order:
        g.add_node(name, fns[name])
    g.set_entry_point(order[0])
    for a, b in zip(order, order[1:]):
        g.add_edge(a, b)
    g.add_edge(order[-1], END)
    return g.compile()


def test_audit_names_reviewer_collapsible_and_keeps_writer() -> None:
    warrant.reset()
    app = warrant.instrument(
        build_graph(),
        node_tools={"retriever": ["arxiv"]},
        tools={"arxiv": "INJECTOR"},
        build_graph=build_graph,
        graph_name="research-brief",
        output_key="text",
    )
    with warrant.session():
        for _ in range(3):
            app.invoke({"text": ""})
        report = warrant.audit()

    by_id = {f.node_id: f for f in report.findings}

    # retriever injects exogenous signal -> keep.
    assert by_id["retriever"].recommendation is Recommendation.KEEP

    # reviewer: reorganizer, ablation value ~0 -> collapse, with a dollar saving.
    reviewer = by_id["reviewer"]
    assert reviewer.recommendation is Recommendation.COLLAPSE
    assert reviewer.ablation_tested is True
    assert reviewer.ablation_value == 0.0
    assert reviewer.ablation_runs == 3
    assert reviewer.projected_savings_per_1k_runs > 0

    # writer: reorganizer structurally, but load-bearing -> keep (not collapsed).
    writer = by_id["writer"]
    assert writer.recommendation is Recommendation.KEEP
    assert writer.ablation_value == 1.0

    assert report.projected_savings_per_1k_runs > 0
    assert len(report.collapsible()) == 1

    # No volume was declared, so no monthly figure is invented anywhere.
    assert report.runs_per_month is None
    assert report.projected_savings_per_month is None
    assert reviewer.dollars_per_month is None
    assert "per 1,000 runs" in report.savings_sentence()


def test_declared_volume_projects_a_monthly_figure() -> None:
    """A monthly number appears only when the caller supplies their traffic."""
    warrant.reset()
    app = warrant.instrument(
        build_graph(),
        node_tools={"retriever": ["arxiv"]},
        tools={"arxiv": "INJECTOR"},
        build_graph=build_graph,
        graph_name="research-brief",
        output_key="text",
    )
    with warrant.session():
        for _ in range(3):
            app.invoke({"text": ""})
        report = warrant.audit(runs_per_month=30_000)

    reviewer = {f.node_id: f for f in report.findings}["reviewer"]
    assert report.runs_per_month == 30_000
    assert reviewer.dollars_per_month is not None
    # 30,000 runs is 30 × 1,000; both figures are rounded for display, so compare
    # loosely — the exact arithmetic is pinned in test_analysis_units.
    assert reviewer.dollars_per_month == pytest.approx(
        reviewer.dollars_per_1k_runs * 30, abs=0.01
    )
    assert report.projected_savings_per_month > 0
    # The word "declared" is load-bearing: the reader must know it's their number.
    assert "declared" in report.savings_sentence()
    warrant.reset()


def test_collapse_confidence_scales_with_evidence() -> None:
    """One clean replay is weak evidence and must not render as near-certainty."""
    warrant.reset()
    app = warrant.instrument(
        build_graph(),
        node_tools={"retriever": ["arxiv"]},
        tools={"arxiv": "INJECTOR"},
        build_graph=build_graph,
        graph_name="research-brief",
        output_key="text",
    )
    with warrant.session():
        app.invoke({"text": ""})
        thin = warrant.audit()
    with warrant.session():
        for _ in range(40):
            app.invoke({"text": ""})
        thick = warrant.audit()
    warrant.reset()

    one = {f.node_id: f for f in thin.findings}["reviewer"]
    many = {f.node_id: f for f in thick.findings}["reviewer"]
    assert one.recommendation is Recommendation.COLLAPSE
    assert many.recommendation is Recommendation.COLLAPSE
    assert one.ablation_runs == 1 and many.ablation_runs == 40
    assert one.confidence < 0.3            # rule of three: 1 run proves little
    assert many.confidence > 0.8           # 40 clean replays is real evidence


def test_audit_degrades_without_build_graph() -> None:
    warrant.reset()
    app = warrant.instrument(
        build_graph(),
        node_tools={"retriever": ["arxiv"]},
        tools={"arxiv": "INJECTOR"},
        graph_name="research-brief",
        output_key="text",
    )
    with warrant.session():
        app.invoke({"text": ""})
        report = warrant.audit()

    # No ablation -> reviewer falls back to the novelty proxy (redundant) at lower confidence.
    reviewer = {f.node_id: f for f in report.findings}["reviewer"]
    assert reviewer.ablation_tested is False
    assert any("Ablation unavailable" in n for n in report.notes)
    assert reviewer.confidence <= 0.5
