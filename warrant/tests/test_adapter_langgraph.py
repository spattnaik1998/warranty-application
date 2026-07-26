"""LangGraph adapter builds a faithful Trace Contract run from a real graph."""

from __future__ import annotations

from typing import TypedDict

import pytest

from warrant.adapters.langgraph import instrument_langgraph, resolve_role
from warrant.tools.registry import ToolRole
from warrant.trace.store import TraceStore

lg = pytest.importorskip("langgraph.graph")


class _S(TypedDict):
    text: str


def _build(disabled: frozenset[str] = frozenset()):
    from langgraph.graph import END, StateGraph

    def retriever(s: _S) -> _S:
        return {"text": s["text"] + "|papers"} if "retriever" not in disabled else {"text": s["text"]}

    def reviewer(s: _S) -> _S:
        # Reorganizer: reads context, adds nothing exogenous.
        return {"text": s["text"] + "|reviewed"} if "reviewer" not in disabled else {"text": s["text"]}

    g = StateGraph(_S)
    g.add_node("retriever", retriever)
    g.add_node("reviewer", reviewer)
    g.set_entry_point("retriever")
    g.add_edge("retriever", "reviewer")
    g.add_edge("reviewer", END)
    return g.compile()


def test_resolve_role_precedence() -> None:
    assert resolve_role("arxiv_search") is ToolRole.INJECTOR
    assert resolve_role("run_tests") is ToolRole.VALIDATOR
    assert resolve_role("mystery") is None
    assert resolve_role("mystery", {"mystery": ToolRole.INJECTOR}) is ToolRole.INJECTOR


def test_adapter_records_nodes_and_tools() -> None:
    store = TraceStore()
    app = instrument_langgraph(
        _build(),
        store,
        graph_name="demo",
        node_tools={"retriever": ["arxiv"]},
        tool_tags={"arxiv": ToolRole.INJECTOR},
        output_key="text",
    )
    final = app.invoke({"text": "x"})

    assert final["text"] == "x|papers|reviewed"
    assert len(store) == 1
    run = store.all()[0]
    assert [n.node_id for n in run.nodes] == ["retriever", "reviewer"]

    retriever = run.node("retriever")
    reviewer = run.node("reviewer")
    assert retriever.has_exogenous_tool is True          # arxiv injector
    assert reviewer.has_exogenous_tool is False           # no tools -> reorganizer candidate
    assert run.final_output == "x|papers|reviewed"
    assert run.total_tokens > 0
    assert run.labels["tokens"] == "estimated"


def test_invoke_runs_graph_once() -> None:
    """The streamed final state must equal a plain invoke — no double execution."""
    store = TraceStore()
    graph = _build()
    app = instrument_langgraph(graph, store, output_key="text")
    assert app.invoke({"text": "y"}) == graph.invoke({"text": "y"})
