"""A small, generic LangGraph research assistant used by the examples.

Deliberately contains one redundant node (``reviewer``) and one load-bearing
reorganizer (``writer``) so the audit has something real to find. This is a
stand-in for *any* customer graph; Warrant needs no knowledge of its internals.

State: ``{"topic", "papers", "draft"}``. Flow::

    retriever(arxiv)  ->  writer  ->  reviewer  ->  END
      injector           reorganizer  reorganizer
                         (load-bearing) (redundant)
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph


class ResearchState(TypedDict, total=False):
    topic: str
    papers: str
    draft: str


def _retriever(state: ResearchState) -> ResearchState:
    # Injects a new exogenous signal (a search result); tagged via node_tools.
    topic = state.get("topic", "unknown")
    return {"papers": f"[arxiv:{topic}] FlashDecode (2604.001), MoE-Router (2604.014)"}


def _writer(state: ResearchState) -> ResearchState:
    # Load-bearing reorganizer: turns retrieved papers into the final draft.
    return {"draft": f"# Brief on {state.get('topic','?')}\n\nPapers: {state.get('papers','')}"}


def _reviewer(state: ResearchState) -> ResearchState:
    # Redundant reorganizer: "reviews" the draft but changes nothing material.
    return {"draft": state.get("draft", "")}


def build_research_graph(disabled: frozenset[str] = frozenset()):
    """Compile the research graph, optionally omitting ``disabled`` nodes.

    The ``disabled`` parameter is what makes the graph *ablatable*: Warrant calls
    ``build_research_graph(frozenset({node}))`` to measure each node's value.
    """
    fns = {"retriever": _retriever, "writer": _writer, "reviewer": _reviewer}
    order = [n for n in ("retriever", "writer", "reviewer") if n not in disabled]

    g = StateGraph(ResearchState)
    for name in order:
        g.add_node(name, fns[name])
    g.set_entry_point(order[0])
    for a, b in zip(order, order[1:]):
        g.add_edge(a, b)
    g.add_edge(order[-1], END)
    return g.compile()


NODE_TOOLS = {"retriever": ["arxiv"]}
