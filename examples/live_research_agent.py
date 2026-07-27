"""A small, *real* async LangGraph research agent — the live-audit artefact.

Unlike ``research_graph`` (which fakes retrieval and generation so tests run
offline), this graph makes genuine calls: arXiv for retrieval and OpenAI for the
worker nodes. It is a stand-in for any real customer graph, and it is
deliberately built with one redundant node so Warrant has something true to find.

Flow (all nodes ``async``, driven by ``ainvoke``/``astream``)::

    search(arxiv)  ->  analyze  ->  synthesize  ->  review  ->  END
      INJECTOR         reorganizer   reorganizer     reorganizer
      (real arXiv)     (load-bearing)(load-bearing)  (REDUNDANT)

Only ``synthesize`` writes the final ``answer``; ``review`` merely appends
commentary and never changes it — so ablating ``review`` leaves the answer
identical and Warrant should flag it COLLAPSE, with the real OpenAI tokens it
burns as the quantified saving. ``analyze`` and ``synthesize`` are load-bearing
(removing either changes the answer), so they stay KEEP.

Requires live mode: ``WARRANT_MOCK=0`` and a valid ``OPENAI_API_KEY``. arXiv
needs no key. Determinism (needed for a clean ablation diff) comes from
``temperature=0`` and a fixed retrieval query per run.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from warrant.config import get_settings

# Node order; the factory omits any disabled node and relinks the chain so the
# graph stays ablatable (Warrant rebuilds it with one node removed at a time).
_ORDER = ["search", "analyze", "synthesize", "review"]

NODE_TOOLS = {"search": ["arxiv"]}  # only 'search' injects an exogenous signal

_MAX_PAPERS = 3
_ABSTRACT_CHARS = 700  # truncate abstracts so prompts stay stable and cheap


class LiveResearchState(TypedDict, total=False):
    query: str
    papers: str
    analysis: str
    answer: str
    review_notes: str
    messages: Annotated[list, add_messages]


def _llm() -> Any:
    """Build a deterministic OpenAI chat model from settings (lazy: needs env)."""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model=settings.worker_model,
        temperature=0,
        max_tokens=500,
        timeout=30,
        max_retries=2,
        api_key=settings.openai_api_key or None,
    )


# Memoize completions by prompt. Baseline runs make real calls (real
# usage_metadata -> measured tokens); ablation then replays the graph with a node
# disabled, and every unchanged upstream prompt is a cache hit -> the replay
# reproduces the *identical* answer. This is what makes ablation deterministic
# despite LLM run-to-run variance: a truly redundant node's removal leaves the
# answer byte-identical, so its delegation value reads a clean 0.
_COMPLETION_CACHE: dict[str, str] = {}


async def _complete(prompt: str) -> Any:
    """Return an AIMessage for ``prompt``, memoized by prompt text."""
    from langchain_core.messages import AIMessage

    if prompt in _COMPLETION_CACHE:
        return AIMessage(content=_COMPLETION_CACHE[prompt])
    msg = await _llm().ainvoke(prompt)
    _COMPLETION_CACHE[prompt] = msg.content
    return msg


def _arxiv_cache_path(query: str) -> Any:
    import hashlib
    from pathlib import Path

    key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return Path(get_settings().output_dir) / "arxiv_cache" / f"{key}.txt"


@lru_cache(maxsize=64)
def _fetch_arxiv(query: str) -> str:
    """Return a compact, stable digest of the top arXiv papers for ``query``.

    Cached two ways: on disk (so a query is fetched from arXiv at most once ever,
    decoupling the audit from arXiv availability / rate limits) and in-process.
    A stable, deterministic retrieval is what lets ablation diff the final answer
    cleanly across replays. Respects arXiv's request-rate etiquette on live fetch.
    """
    cache_file = _arxiv_cache_path(query)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    import arxiv

    client = arxiv.Client(page_size=_MAX_PAPERS, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=_MAX_PAPERS,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    lines: list[str] = []
    for r in client.results(search):
        abstract = " ".join(r.summary.split())[:_ABSTRACT_CHARS]
        lines.append(f"- {r.title} ({r.get_short_id()}): {abstract}")
    result = "\n".join(lines) if lines else "(no papers found)"

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(result, encoding="utf-8")
    return result


async def _search(state: LiveResearchState) -> LiveResearchState:
    """INJECTOR: retrieve real papers from arXiv (new exogenous signal)."""
    query = state.get("query", "")
    papers = await asyncio.to_thread(_fetch_arxiv, query)
    return {"papers": papers}


async def _analyze(state: LiveResearchState) -> LiveResearchState:
    """Load-bearing reorganizer: distil the retrieved abstracts into key findings."""
    prompt = (
        "You are an ML research analyst. From the paper list below, extract the 3 "
        "most important technical findings as terse bullet points. Papers:\n\n"
        f"{state.get('papers', '')}"
    )
    msg = await _complete(prompt)
    return {"analysis": msg.content, "messages": [msg]}


async def _synthesize(state: LiveResearchState) -> LiveResearchState:
    """Load-bearing reorganizer: write the final briefing (sets the answer)."""
    prompt = (
        "Write a concise technical briefing (max 150 words) for an ML engineer, "
        f"on the query '{state.get('query', '')}', grounded ONLY in these findings:\n\n"
        f"{state.get('analysis', '')}"
    )
    msg = await _complete(prompt)
    return {"answer": msg.content, "messages": [msg]}


async def _review(state: LiveResearchState) -> LiveResearchState:
    """REDUNDANT reorganizer: 'reviews' the briefing but never changes the answer.

    It re-reads context the system already has and adds no exogenous signal, so it
    is dominated by the orchestrator. Its output goes to ``review_notes`` — the
    final ``answer`` is untouched — which is exactly why ablation finds it
    collapsible even though it burns real tokens.
    """
    prompt = (
        "Review the following briefing for tone and clarity and reply with a short "
        "approval note. Do NOT rewrite it.\n\n"
        f"{state.get('answer', '')}"
    )
    msg = await _complete(prompt)
    return {"review_notes": msg.content, "messages": [msg]}


_FNS = {"search": _search, "analyze": _analyze, "synthesize": _synthesize, "review": _review}


def build_live_graph(disabled: frozenset[str] = frozenset()):
    """Compile the live research graph, optionally omitting ``disabled`` nodes.

    Passing a disabled set is what makes the graph ablatable: Warrant calls
    ``build_live_graph(frozenset({node}))`` to measure each node's delegation value.
    """
    order = [n for n in _ORDER if n not in disabled]
    if not order:
        raise ValueError("cannot disable every node")

    g = StateGraph(LiveResearchState)
    for name in order:
        g.add_node(name, _FNS[name])
    g.set_entry_point(order[0])
    for a, b in zip(order, order[1:]):
        g.add_edge(a, b)
    g.add_edge(order[-1], END)
    return g.compile()
