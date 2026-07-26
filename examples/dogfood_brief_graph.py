"""Dogfood: wrap Warrant's own briefing pipeline as a LangGraph app and audit it.

The generic SDK is pointed at the hand-built demo pipeline. If the abstractions
are right, the audit must *independently rediscover* what the governed
orchestrator was designed around: the ``compose`` stage injects no exogenous
signal and is therefore a REORGANIZER — the same classification that made the
orchestrator do composition centrally rather than delegate it.

Runs fully offline under ``WARRANT_MOCK=1``.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from warrant.pipeline.steps import (
    PipelineContext,
    compose_sections,
    decision_keys,
    discover_target,
    elicit_key,
    fetch_paper,
    new_context,
    read_paper,
)
from warrant.schemas.belief import AdmissibilityClass
from warrant.schemas.tasks import BriefRequest


class BriefState(TypedDict, total=False):
    request: BriefRequest
    ctx: PipelineContext
    arxiv_id: str
    markdown: str


def _discover(s: BriefState) -> BriefState:
    ctx = new_context(s["request"])
    return {"ctx": ctx, "arxiv_id": discover_target(ctx)}


def _fetch(s: BriefState) -> BriefState:
    fetch_paper(s["ctx"], s["arxiv_id"])
    return {"arxiv_id": s["arxiv_id"]}


def _read(s: BriefState) -> BriefState:
    read_paper(s["ctx"], s["arxiv_id"])
    return {"arxiv_id": s["arxiv_id"]}


def _compose(s: BriefState) -> BriefState:
    # Reorganization only: elicits beliefs from already-gathered text and lays
    # out the briefing. Calls no exogenous tool -> a REORGANIZER by construction.
    ctx = s["ctx"]
    dks = decision_keys(ctx)
    text = ctx.artifacts.get("paper_text", "")
    for dk in dks:
        elicit_key(ctx, dk, text, delegation_type=dk.key, cls=AdmissibilityClass.VALIDATOR)
    sections = compose_sections(ctx, dks)
    return {"markdown": "\n\n".join(f"## {sec.heading}\n{sec.body}" for sec in sections)}


# discover/fetch/read call exogenous tools; compose calls none.
NODE_TOOLS = {
    "discover": ["youtube_latest"],
    "fetch": ["arxiv_fetch"],
    "read": ["pdf_read"],
    "compose": [],
}
_FNS = {"discover": _discover, "fetch": _fetch, "read": _read, "compose": _compose}


def build_brief_graph(disabled: frozenset[str] = frozenset()):
    order = [n for n in ("discover", "fetch", "read", "compose") if n not in disabled]
    g = StateGraph(BriefState)
    for name in order:
        g.add_node(name, _FNS[name])
    g.set_entry_point(order[0])
    for a, b in zip(order, order[1:]):
        g.add_edge(a, b)
    g.add_edge(order[-1], END)
    return g.compile()
