"""Warrant — a delegation-economics SDK for multi-agent LLM systems.

You wrap your existing agent graph, run it, and Warrant tells you *which of your
agents should not exist* — and proves the token/latency/dollar savings from
collapsing or pruning them. It operationalizes the reliability-limits theorem of
Ao, Gao & Simchi-Levi (arXiv:2603.26993): without a new *exogenous* signal, a
delegated multi-agent network is decision-theoretically dominated by a single
centralized Bayes decision-maker, and extra relay hops only add posterior
distortion (the "telephone game").

Quickstart::

    import warrant

    app = warrant.instrument(compiled_graph, node_tools={"retriever": ["arxiv"]})
    app.invoke(inputs)
    report = warrant.audit()
    report.to_html("out/audit.html")

See ``PRODUCT_DIRECTION.md`` for the product thesis and the capability ladder.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping

from warrant.schemas.belief import Posterior
from warrant.tools.registry import ToolRole
from warrant.trace.contract import DecisionPoint, RunTrace
from warrant.trace.store import TraceStore

# Silence a noisy pending-deprecation warning emitted by langgraph's serializer
# when the adapter lazily imports it; it does not affect our usage.
warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change",
)

__version__ = "0.1.0"

# --------------------------------------------------------------------------- #
# Module-level session state. A single active store + the app under audit.
# --------------------------------------------------------------------------- #
_STORE: TraceStore = TraceStore()
_CURRENT_APP: Any | None = None
_TOOL_TAGS: dict[str, ToolRole] = {}
_PENDING_DECISIONS: list[tuple[str, DecisionPoint]] = []


def instrument(
    app: Any,
    *,
    node_tools: Mapping[str, list[str]] | None = None,
    tools: Mapping[str, ToolRole] | None = None,
    build_graph: Callable[[frozenset[str]], Any] | None = None,
    mode: str = "observe",
    graph_name: str = "",
    output_key: str | None = None,
    store: TraceStore | None = None,
) -> Any:
    """Wrap a compiled LangGraph app for auditing and make it the active app.

    Args:
        app: a compiled LangGraph graph (anything with ``.stream``/``.invoke``).
        node_tools: ``{node_id: [tool_name, ...]}`` — which tools each node uses.
            Needed for the zero-annotation structural audit (capability 1).
        tools: optional explicit ``{tool_name: ToolRole}`` tags.
        build_graph: ``build_graph(disabled_nodes) -> compiled graph`` factory;
            required to unlock ablation-based delegation value (capability 2).
        mode: ``"observe"`` (default, non-invasive) or ``"govern"`` (post-MVP).
        graph_name / output_key: display name and the state key holding the
            final answer (used for ablation diffing and the report).
        store: override the active trace store (defaults to the module store).

    Returns:
        An :class:`~warrant.adapters.langgraph.InstrumentedApp` proxying ``app``.
    """
    from warrant.adapters.langgraph import instrument_langgraph

    global _CURRENT_APP, _STORE
    if store is not None:
        _STORE = store
    if tools:
        _TOOL_TAGS.update(tools)
    _CURRENT_APP = instrument_langgraph(
        app,
        _STORE,
        graph_name=graph_name,
        node_tools=node_tools,
        tool_tags=_TOOL_TAGS,
        build_graph=build_graph,
        output_key=output_key,
        mode=mode,
    )
    return _CURRENT_APP


@contextmanager
def session(store: TraceStore | None = None) -> Iterator[TraceStore]:
    """Scope a set of runs. Clears prior traces so ``audit()`` covers only this scope."""
    global _STORE
    active = store or _STORE
    active.clear()
    _PENDING_DECISIONS.clear()
    prev, _STORE = _STORE, active
    if _CURRENT_APP is not None:
        _CURRENT_APP._store = active
    try:
        yield active
    finally:
        _STORE = prev
        if _CURRENT_APP is not None:
            _CURRENT_APP._store = prev


def tool_tag(name: str, role: str | ToolRole) -> None:
    """Tag a tool name with a :class:`ToolRole` (INJECTOR / VALIDATOR / TRANSFORM)."""
    resolved = role if isinstance(role, ToolRole) else ToolRole(str(role).upper())
    _TOOL_TAGS[name] = resolved
    if _CURRENT_APP is not None:
        _CURRENT_APP.tool_tags[name] = resolved


def decision(
    key: str,
    options: list[str] | None = None,
    posterior: Posterior | None = None,
    *,
    node: str,
    claim: str = "",
) -> None:
    """Annotate a belief checkpoint at a node (capability level 4, opt-in).

    Attaches to the named node in the most recent recorded run if present,
    otherwise buffers and attaches at the next :func:`audit`. Supplying decision
    points unlocks the rigorous posterior-distortion analysis and the
    matched-condition ledger; without them the audit uses structural + ablation
    + novelty signals only.
    """
    dp = DecisionPoint(key=key, options=options or [], posterior=posterior, claim=claim)
    runs = _STORE.all()
    if runs and runs[-1].node(node) is not None:
        runs[-1].node(node).decision = dp
    else:
        _PENDING_DECISIONS.append((node, dp))


def _drain_pending_decisions() -> None:
    """Attach buffered decisions to their nodes in the latest run."""
    if not _PENDING_DECISIONS:
        return
    runs = _STORE.all()
    if not runs:
        return
    latest = runs[-1]
    remaining: list[tuple[str, DecisionPoint]] = []
    for node_id, dp in _PENDING_DECISIONS:
        target = latest.node(node_id)
        if target is not None:
            target.decision = dp
        else:
            remaining.append((node_id, dp))
    _PENDING_DECISIONS[:] = remaining


def audit(store: TraceStore | None = None) -> Any:
    """Analyze the recorded runs and return an :class:`AuditReport`."""
    from warrant.analysis import run_audit

    _drain_pending_decisions()
    return run_audit((store or _STORE).all(), app=_CURRENT_APP)


def export() -> list[RunTrace]:
    """Cloud ingestion seam: return the active traces (warrant-cloud uploads these)."""
    return _STORE.all()


def reset() -> None:
    """Clear all session state. Primarily for tests."""
    global _CURRENT_APP
    _STORE.clear()
    _TOOL_TAGS.clear()
    _PENDING_DECISIONS.clear()
    _CURRENT_APP = None


__all__ = [
    "instrument",
    "session",
    "tool_tag",
    "decision",
    "audit",
    "export",
    "reset",
    "Posterior",
    "ToolRole",
    "__version__",
]
