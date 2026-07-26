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
    assert reviewer.projected_savings_per_month > 0

    # writer: reorganizer structurally, but load-bearing -> keep (not collapsed).
    writer = by_id["writer"]
    assert writer.recommendation is Recommendation.KEEP
    assert writer.ablation_value == 1.0

    assert report.projected_savings_per_month > 0
    assert len(report.collapsible()) == 1


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
