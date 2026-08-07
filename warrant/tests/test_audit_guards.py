"""The audit must not report a confident verdict on evidence it doesn't have.

Three ways a first-time user gets a wrong answer silently, and the guard for each:

* no ``node_tools`` declared and none observed → *every* node looks like a
  reorganizer, so the whole table reads COLLAPSE;
* no ``output_key`` declared → ablation diffs the whole state, so *every* node
  looks load-bearing and nothing is ever collapsible;
* a nondeterministic graph → the diff measures sampling noise, not delegation.

Each must produce a note and a capped confidence rather than a clean-looking
verdict. Also covered: tool calls the framework already reports are picked up
without any declaration at all.
"""

from __future__ import annotations

from typing import TypedDict

import pytest

import warrant
from warrant.analysis.report import Recommendation

pytest.importorskip("langgraph.graph")


class _ToolMessage:
    """Stands in for a LangChain ToolMessage."""

    type = "tool"

    def __init__(self, name: str) -> None:
        self.name = name
        self.content = f"result of {name}"


class _S(TypedDict):
    text: str
    messages: list


def _plain_graph(disabled: frozenset[str] = frozenset()):
    """retriever -> writer -> reviewer, with no tool metadata of any kind."""
    from langgraph.graph import END, StateGraph

    def retriever(s: _S) -> _S:
        return {"text": "papers: A, B, C"}

    def writer(s: _S) -> _S:
        return {"text": "BRIEF >> " + s["text"]}

    def reviewer(s: _S) -> _S:
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


def test_observed_tool_calls_need_no_declaration() -> None:
    """A ToolMessage in the state is proof the tool ran — no node_tools needed."""
    from langgraph.graph import END, StateGraph

    def search(s: _S) -> _S:
        return {"messages": [_ToolMessage("web_search")], "text": "found"}

    def summarize(s: _S) -> _S:
        return {"text": "summary of " + s["text"]}

    g = StateGraph(_S)
    g.add_node("search", search)
    g.add_node("summarize", summarize)
    g.set_entry_point("search")
    g.add_edge("search", "summarize")
    g.add_edge("summarize", END)

    warrant.reset()
    # Deliberately no node_tools: the audit must find the tool on its own.
    app = warrant.instrument(g.compile(), graph_name="observed", output_key="text")
    with warrant.session():
        app.invoke({"text": "", "messages": []})
        report = warrant.audit()
    warrant.reset()

    by_id = {f.node_id: f for f in report.findings}
    assert by_id["search"].tools == ["web_search"]
    assert by_id["search"].recommendation is Recommendation.KEEP     # injector
    assert by_id["summarize"].tools == []                             # nothing observed
    # Tools *were* observed somewhere, so the "no tools at all" guard stays quiet.
    assert not any("No tool calls were declared or observed" in n for n in report.notes)


def test_no_tools_anywhere_is_flagged_and_capped() -> None:
    """With no tool evidence every node is a reorganizer — say so, don't assert it."""
    warrant.reset()
    app = warrant.instrument(
        _plain_graph(), build_graph=_plain_graph, graph_name="blind", output_key="text"
    )
    with warrant.session():
        for _ in range(5):
            app.invoke({"text": "", "messages": []})
        report = warrant.audit()
    warrant.reset()

    assert report.notes[0].startswith("No tool calls were declared or observed")
    assert all(f.admissibility.value == "REORGANIZER" for f in report.findings)
    # The verdicts may still be right, but none may claim to be well-evidenced.
    assert all(f.confidence <= 0.5 for f in report.findings)


def test_missing_output_key_is_flagged_and_capped() -> None:
    """Diffing the whole state biases everything to KEEP; the report must say so."""
    warrant.reset()
    app = warrant.instrument(
        _plain_graph(),
        node_tools={"retriever": ["arxiv"]},
        tools={"arxiv": "INJECTOR"},
        build_graph=_plain_graph,
        graph_name="no-key",
    )
    with warrant.session():
        for _ in range(5):
            app.invoke({"text": "", "messages": []})
        report = warrant.audit()
    warrant.reset()

    assert any("No output_key was declared" in n for n in report.notes)
    reviewer = {f.node_id: f for f in report.findings}["reviewer"]
    assert reviewer.confidence <= 0.6


def test_nondeterministic_graph_caps_confidence() -> None:
    """A graph that can't reproduce itself makes every ablation diff meaningless."""
    from itertools import count

    from langgraph.graph import END, StateGraph

    ticker = count()

    def build(disabled: frozenset[str] = frozenset()):
        def retriever(s: _S) -> _S:
            return {"text": "papers"}

        def writer(s: _S) -> _S:
            # Output changes every call: sampling noise, not delegation value.
            return {"text": f"{s['text']}|draft-{next(ticker)}"}

        def reviewer(s: _S) -> _S:
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

    warrant.reset()
    app = warrant.instrument(
        build(),
        node_tools={"retriever": ["arxiv"]},
        tools={"arxiv": "INJECTOR"},
        build_graph=build,
        graph_name="noisy",
        output_key="text",
    )
    with warrant.session():
        for _ in range(3):
            app.invoke({"text": "", "messages": []})
        report = warrant.audit()
    warrant.reset()

    assert any("did not reproduce its own recorded output" in n for n in report.notes)
    reorganizers = [f for f in report.findings if f.admissibility.value == "REORGANIZER"]
    assert reorganizers
    assert all(f.confidence <= 0.4 for f in reorganizers)
