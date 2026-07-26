"""Dogfood regression: the generic SDK rediscovers the governed design.

Instrumenting Warrant's own briefing pipeline must classify ``compose`` as a
REORGANIZER — the same conclusion the hand-built governed orchestrator was
designed around — without any pipeline-specific knowledge in the audit engine.
"""

from __future__ import annotations

import pytest

import warrant
from warrant.schemas.belief import AdmissibilityClass
from warrant.schemas.tasks import BriefRequest

pytest.importorskip("langgraph.graph")


def test_audit_rediscovers_compose_reorganizer() -> None:
    from examples.dogfood_brief_graph import NODE_TOOLS, build_brief_graph

    warrant.reset()
    app = warrant.instrument(
        build_brief_graph(),
        node_tools=NODE_TOOLS,
        build_graph=build_brief_graph,
        graph_name="warrant-briefing",
        output_key="markdown",
    )
    with warrant.session():
        app.invoke({"request": BriefRequest(arxiv_id="2603.26993")})
        report = warrant.audit()

    by_id = {f.node_id: f for f in report.findings}
    assert by_id["compose"].admissibility is AdmissibilityClass.REORGANIZER
    # The exogenous stages earn their warrant.
    for node in ("discover", "fetch", "read"):
        assert by_id[node].admissibility is AdmissibilityClass.INJECTOR
