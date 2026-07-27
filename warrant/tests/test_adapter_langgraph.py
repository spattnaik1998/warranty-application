"""LangGraph adapter builds a faithful Trace Contract run from a real graph."""

from __future__ import annotations

import asyncio
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


def test_async_adapter_records_async_graph() -> None:
    """Async graphs (async nodes, driven by ainvoke/astream) record correctly."""
    from langgraph.graph import END, StateGraph

    class _AS(TypedDict):
        messages: list

    async def retriever(s: _AS) -> _AS:
        return {"messages": s["messages"] + [_FakeMessage("papers", 400)]}

    async def writer(s: _AS) -> _AS:
        return {"messages": s["messages"] + [_FakeMessage("essay", 250)]}

    g = StateGraph(_AS)
    g.add_node("retriever", retriever)
    g.add_node("writer", writer)
    g.set_entry_point("retriever")
    g.add_edge("retriever", "writer")
    g.add_edge("writer", END)

    store = TraceStore()
    app = instrument_langgraph(
        g.compile(), store, graph_name="async-demo", node_tools={"retriever": ["arxiv"]}
    )

    final = asyncio.run(app.ainvoke({"messages": []}))
    assert len(final["messages"]) == 2

    run = store.all()[0]
    assert [n.node_id for n in run.nodes] == ["retriever", "writer"]
    assert run.node("retriever").outcome.tokens == 400   # measured, not estimated
    assert run.node("writer").outcome.tokens == 250      # echo of retriever counted once
    assert run.labels["tokens"] == "measured"


def test_async_ablation_discriminates_from_running_loop() -> None:
    """End-to-end async audit: ablation must run correctly even when audit() is
    called from inside a running event loop (auditing at the end of an async run)."""
    import warrant
    from langgraph.graph import END, StateGraph

    class _RS(TypedDict):
        messages: list
        answer: str

    def build(disabled: frozenset[str] = frozenset()):
        async def retriever(s: _RS) -> _RS:
            return {} if "retriever" in disabled else {"messages": s["messages"] + [_FakeMessage("papers", 600)]}

        async def writer(s: _RS) -> _RS:  # load-bearing: only this sets the answer
            return {} if "writer" in disabled else {"answer": "ESSAY", "messages": s["messages"] + [_FakeMessage("draft", 400)]}

        async def reviewer(s: _RS) -> _RS:  # redundant: never changes the answer
            return {} if "reviewer" in disabled else {"messages": s["messages"] + [_FakeMessage("lgtm", 200)]}

        g = StateGraph(_RS)
        for name, fn in [("retriever", retriever), ("writer", writer), ("reviewer", reviewer)]:
            g.add_node(name, fn)
        g.set_entry_point("retriever")
        g.add_edge("retriever", "writer")
        g.add_edge("writer", "reviewer")
        g.add_edge("reviewer", END)
        return g.compile()

    async def main():
        warrant.reset()
        app = warrant.instrument(
            build(), node_tools={"retriever": ["arxiv"]}, build_graph=build,
            output_key="answer", graph_name="async-research",
        )
        with warrant.session():
            for _ in range(3):
                await app.ainvoke({"messages": [], "answer": ""})
            return warrant.audit()

    report = asyncio.run(main())
    warrant.reset()

    by_id = {f.node_id: f for f in report.findings}
    assert by_id["writer"].ablation_value == 1.0        # load-bearing -> KEEP
    assert by_id["reviewer"].ablation_value == 0.0       # redundant   -> COLLAPSE
    assert by_id["reviewer"].recommendation.value == "COLLAPSE"
    assert report.projected_savings_per_month > 0


def test_invoke_runs_graph_once() -> None:
    """The streamed final state must equal a plain invoke — no double execution."""
    store = TraceStore()
    graph = _build()
    app = instrument_langgraph(graph, store, output_key="text")
    assert app.invoke({"text": "y"}) == graph.invoke({"text": "y"})
