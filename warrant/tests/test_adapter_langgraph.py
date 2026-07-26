"""LangGraph adapter builds a faithful Trace Contract run from a real graph."""

from __future__ import annotations

from typing import TypedDict

import pytest

from warrant.adapters.langgraph import _extract_usage, instrument_langgraph, resolve_role
from warrant.tools.registry import ToolRole
from warrant.trace.store import TraceStore


class _FakeMessage:
    """Stands in for a LangChain AIMessage carrying real usage metadata."""

    def __init__(self, content: str, total_tokens: int) -> None:
        self.content = content
        self.usage_metadata = {
            "input_tokens": total_tokens // 2,
            "output_tokens": total_tokens - total_tokens // 2,
            "total_tokens": total_tokens,
        }

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


def test_extract_usage_reads_real_metadata() -> None:
    # Nested in a messages list, as LangGraph state typically carries them.
    update = {"messages": [_FakeMessage("hi", 120), _FakeMessage("there", 30)]}
    total, found = _extract_usage(update)
    assert (total, found) == (150, True)

    # A summed input/output with no total_tokens still resolves.
    partial = type("M", (), {"usage_metadata": {"input_tokens": 10, "output_tokens": 7}})()
    assert _extract_usage({"m": partial}) == (17, True)

    # Nothing measurable -> caller must fall back to estimation.
    assert _extract_usage({"text": "no usage here"}) == (0, False)


def test_extract_usage_counts_echoed_message_once() -> None:
    # A shared ``seen`` set means a message present in two nodes' states
    # (message-list state without a reducer) is attributed only to the first.
    m = _FakeMessage("shared", 500)
    seen: set[int] = set()
    assert _extract_usage({"messages": [m]}, seen) == (500, True)
    assert _extract_usage({"messages": [m, _FakeMessage("new", 40)]}, seen) == (40, True)


def test_adapter_uses_measured_tokens_when_available() -> None:
    from langgraph.graph import END, StateGraph

    class _MS(TypedDict):
        messages: list

    def worker(s: _MS) -> _MS:
        return {"messages": s["messages"] + [_FakeMessage("answer", 200)]}

    g = StateGraph(_MS)
    g.add_node("worker", worker)
    g.set_entry_point("worker")
    g.add_edge("worker", END)

    store = TraceStore()
    app = instrument_langgraph(g.compile(), store, graph_name="measured")
    app.invoke({"messages": []})

    run = store.all()[0]
    assert run.node("worker").outcome.tokens == 200   # real usage, not len/4
    assert run.labels["tokens"] == "measured"


def test_invoke_runs_graph_once() -> None:
    """The streamed final state must equal a plain invoke — no double execution."""
    store = TraceStore()
    graph = _build()
    app = instrument_langgraph(graph, store, output_key="text")
    assert app.invoke({"text": "y"}) == graph.invoke({"text": "y"})
