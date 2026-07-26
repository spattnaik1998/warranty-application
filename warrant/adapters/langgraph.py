"""LangGraph adapter — deep, per the LangGraph-first strategy.

Wraps a *compiled* LangGraph app. On invoke it drives the graph with
``stream_mode=["updates", "values"]`` so it sees, in one pass, every node
execution (``updates``) and the accumulated final state (``values``), and
emits a :class:`~warrant.trace.contract.RunTrace`.

The adapter never inspects LangGraph internals beyond this public streaming
contract, so it degrades gracefully across versions and is trivially faked in
tests (any object exposing ``.stream(input, config, stream_mode=...)`` and
``.invoke`` works).

Tool attribution: a generic graph node is opaque about which tools it called,
so the caller declares ``node_tools={node_id: [tool_name, ...]}``. Each tool
name is resolved to a :class:`ToolRole` via, in order: an explicit
``warrant.tool_tag``, the process tool ``REGISTRY``, then a name heuristic.

Token counts come from LangChain's real ``usage_metadata`` when a node's output
carries it (the modern ``AIMessage`` field, or a legacy
``response_metadata['token_usage']``); only when no measurable usage is present
does the adapter fall back to a length estimate. Each run is labelled
``tokens=measured`` / ``mixed`` / ``estimated`` accordingly, so no analyzer — and
no dollar figure — mistakes an estimate for a measurement.
"""

from __future__ import annotations

import time
import uuid
from math import ceil
from typing import Any, Callable, Mapping

from warrant.logging_setup import get_logger, log_event
from warrant.tools.registry import REGISTRY, ToolRole
from warrant.trace.contract import (
    NodeRun,
    Outcome,
    RunStatus,
    RunTrace,
    ToolCallRecord,
)

log = get_logger("warrant.adapters.langgraph")

# Substring heuristics for auto-tagging a tool whose role is otherwise unknown.
_INJECTOR_HINTS = ("search", "arxiv", "retriev", "fetch", "web", "http", "youtube", "pdf", "sql", "db", "api", "lookup")
_VALIDATOR_HINTS = ("test", "verify", "validate", "check", "lint", "exec", "run_code", "factcheck")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for when no usage metadata exists."""
    return ceil(len(text) / 4) if text else 0


def _extract_usage(value: Any, seen: set[int] | None = None) -> tuple[int, bool]:
    """Sum real LLM token usage in a node update, if the framework exposes it.

    Walks the update for LangChain message-like objects carrying real usage:
    the modern ``AIMessage.usage_metadata`` (``{"input_tokens", "output_tokens",
    "total_tokens"}``) or a legacy ``response_metadata["token_usage"]``. Returns
    ``(total_tokens, found)``; ``found`` is False when nothing measurable is
    present, so the caller falls back to the length estimate and the run is
    labelled honestly.

    ``seen`` may be shared across a run's nodes so a message echoed in a later
    node's state (e.g. message-list state without an ``add_messages`` reducer) is
    counted once, at the node that produced it — not re-attributed downstream.
    """
    total = 0
    found = False
    counted: set[int] = seen if seen is not None else set()  # usage objects already tallied
    local: set[int] = set()  # per-call cycle guard on containers (ids get recycled across calls)

    def _usage_total(meta: Any) -> int | None:
        if isinstance(meta, Mapping):
            tokens = meta.get("total_tokens")
            if tokens is None:
                given = meta.get("input_tokens"), meta.get("output_tokens")
                if any(t is not None for t in given):
                    tokens = sum(int(t) for t in given if t is not None)
            if tokens is not None:
                return int(tokens)
        return None

    def walk(obj: Any) -> None:
        nonlocal total, found
        oid = id(obj)
        if oid in local:
            return
        local.add(oid)

        direct = _usage_total(getattr(obj, "usage_metadata", None))
        if direct is None:
            rm = getattr(obj, "response_metadata", None)
            if isinstance(rm, Mapping):
                direct = _usage_total(rm.get("token_usage") or rm.get("usage"))
        if direct is not None:
            # Usage is present, so this update is measured — but only add tokens
            # the first time we see this specific message (echoes count once).
            found = True
            if oid not in counted:
                counted.add(oid)
                total += direct
            return  # counted this message; don't recurse into its fields

        if isinstance(obj, Mapping):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(value)
    return total, found


def _stringify(value: Any) -> str:
    """Deterministic, human-readable rendering of a node update / final state."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "\n".join(f"{k}: {_stringify(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(v) for v in value)
    return str(value)


def resolve_role(
    tool_name: str,
    tags: Mapping[str, ToolRole] | None = None,
) -> ToolRole | None:
    """Resolve a tool name to a role: explicit tag > registry > heuristic > None."""
    if tags and tool_name in tags:
        return tags[tool_name]
    if REGISTRY.has(tool_name):
        return REGISTRY.get(tool_name).role
    lowered = tool_name.lower()
    if any(h in lowered for h in _VALIDATOR_HINTS):
        return ToolRole.VALIDATOR
    if any(h in lowered for h in _INJECTOR_HINTS):
        return ToolRole.INJECTOR
    return None


class InstrumentedApp:
    """A LangGraph app wrapped for observation.

    Proxies ``invoke``/``stream`` to the underlying graph while recording a
    Trace Contract run into the bound store. In observe mode it changes no
    behaviour; it only measures.
    """

    def __init__(
        self,
        app: Any,
        store: Any,
        *,
        graph_name: str = "",
        node_tools: Mapping[str, list[str]] | None = None,
        tool_tags: Mapping[str, ToolRole] | None = None,
        build_graph: Callable[[frozenset[str]], Any] | None = None,
        output_key: str | None = None,
        mode: str = "observe",
    ) -> None:
        self._app = app
        self._store = store
        self.graph_name = graph_name or getattr(app, "name", "") or "graph"
        self.node_tools = dict(node_tools or {})
        self.tool_tags = dict(tool_tags or {})
        self.build_graph = build_graph
        self.output_key = output_key
        self.mode = mode
        # Inputs are retained in-memory (not in the serializable trace) so
        # ablation can replay them against a rebuilt graph.
        self.replays: list[tuple[str, Any, dict | None]] = []

    # -- attribute passthrough so the wrapper is a drop-in for most callers ---
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial proxy
        return getattr(self._app, item)

    def _tool_calls_for(self, node_id: str) -> list[ToolCallRecord]:
        records: list[ToolCallRecord] = []
        for name in self.node_tools.get(node_id, []):
            role = resolve_role(name, self.tool_tags)
            spec = REGISTRY.get(name) if REGISTRY.has(name) else None
            records.append(
                ToolCallRecord(
                    name=name,
                    role=role,
                    signal_source=spec.signal_source if spec else None,
                )
            )
        return records

    def _final_output(self, state: Any) -> str:
        if self.output_key and isinstance(state, Mapping) and self.output_key in state:
            return _stringify(state[self.output_key])
        return _stringify(state)

    def run(
        self,
        graph: Any,
        input: Any,
        config: dict | None = None,
        *,
        labels: dict[str, str] | None = None,
        record: bool = True,
    ) -> RunTrace:
        """Execute ``graph`` on ``input`` and return the resulting RunTrace.

        Used both for the primary instrumented run and by ablation (which
        passes a rebuilt graph and ``record=False``).
        """
        run, _ = self._execute(graph, input, config, labels=labels, record=record)
        return run

    def _execute(
        self,
        graph: Any,
        input: Any,
        config: dict | None = None,
        *,
        labels: dict[str, str] | None = None,
        record: bool = True,
    ) -> tuple[RunTrace, Any]:
        """Stream ``graph`` once, building the trace and capturing final state."""
        run = RunTrace(
            run_id=uuid.uuid4().hex[:12],
            graph_name=self.graph_name,
            mode=self.mode,
            labels={"tokens": "estimated", **(labels or {})},
        )
        final_state: Any = None
        prev = time.perf_counter()
        measured_nodes = 0
        node_count = 0
        usage_seen: set[int] = set()  # shared so echoed messages count once
        try:
            for mode, chunk in graph.stream(input, config, stream_mode=["updates", "values"]):
                if mode == "values":
                    final_state = chunk
                    continue
                # mode == "updates": {node_id: partial_state}
                now = time.perf_counter()
                latency_ms = int((now - prev) * 1000)
                prev = now
                for node_id, update in chunk.items():
                    text = _stringify(update)
                    real_tokens, measured = _extract_usage(update, usage_seen)
                    node_count += 1
                    measured_nodes += int(measured)
                    run.add_node(
                        NodeRun(
                            node_id=node_id,
                            tool_calls=self._tool_calls_for(node_id),
                            outcome=Outcome(
                                tokens=real_tokens if measured else _estimate_tokens(text),
                                latency_ms=latency_ms,
                                output_text=text,
                            ),
                        )
                    )
        except Exception as exc:  # surface, never swallow (CLAUDE.md)
            run.status = RunStatus.ERROR
            run.labels["error"] = str(exc)
            log_event(log, "run failed", stage="adapter.run", graph=self.graph_name, status="error", error=str(exc))
            if record:
                self._store.add(run.finalize())
            raise

        # Label token provenance honestly: measured (all nodes had real usage),
        # mixed (some did), or the default estimated (none did). Downstream $
        # figures read this so an estimate is never presented as a measurement.
        if "tokens" not in (labels or {}):
            if node_count and measured_nodes == node_count:
                run.labels["tokens"] = "measured"
            elif measured_nodes:
                run.labels["tokens"] = "mixed"

        run.finalize(self._final_output(final_state))
        log_event(
            log,
            "run recorded" if record else "replay complete",
            stage="adapter.run",
            graph=self.graph_name,
            status="ok",
            nodes=len(run.nodes),
            tokens=run.total_tokens,
            recorded=record,
        )
        if record:
            self._store.add(run)
            self.replays.append((run.run_id, input, config))
        return run, final_state

    def replay_output(self, graph: Any, input: Any, config: dict | None = None) -> str:
        """Run ``graph`` on ``input`` without recording; return its final output text.

        Used by ablation to compare a node-disabled graph against the baseline.
        """
        _, final_state = self._execute(graph, input, config, record=False)
        return self._final_output(final_state)

    def invoke(self, input: Any, config: dict | None = None, **_: Any) -> Any:
        """Drop-in replacement for ``graph.invoke``; records the run as a side effect.

        Runs the graph exactly once: the streamed ``values`` chunks reconstruct
        the same final state ``graph.invoke`` would return.
        """
        _, final_state = self._execute(self._app, input, config)
        return final_state

    def stream(self, input: Any, config: dict | None = None, **kw: Any) -> Any:
        """Passthrough stream (not recorded); prefer ``invoke`` for auditing."""
        return self._app.stream(input, config, **kw)


def instrument_langgraph(app: Any, store: Any, **kwargs: Any) -> InstrumentedApp:
    """Wrap a compiled LangGraph app for observation. See :class:`InstrumentedApp`."""
    return InstrumentedApp(app, store, **kwargs)


__all__ = ["InstrumentedApp", "instrument_langgraph", "resolve_role"]
