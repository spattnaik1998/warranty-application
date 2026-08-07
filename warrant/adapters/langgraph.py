"""LangGraph adapter — deep, per the LangGraph-first strategy.

Wraps a *compiled* LangGraph app. On invoke it drives the graph with
``stream_mode=["updates", "values"]`` so it sees, in one pass, every node
execution (``updates``) and the accumulated final state (``values``), and
emits a :class:`~warrant.trace.contract.RunTrace`.

The adapter never inspects LangGraph internals beyond this public streaming
contract, so it degrades gracefully across versions and is trivially faked in
tests (any object exposing ``.stream(input, config, stream_mode=...)`` and
``.invoke`` works).

Tool attribution has two sources, unioned. Tool activity LangChain actually
reports — ``ToolMessage`` results and ``AIMessage.tool_calls`` in the streamed
update — is *observed*, so a standard tool-calling graph needs no annotation at
all. Anything the framework doesn't surface (a node that calls an API directly)
is *declared* by the caller as ``node_tools={node_id: [tool_name, ...]}``. Each
tool name is resolved to a :class:`ToolRole` via, in order: an explicit
``warrant.tool_tag``, the process tool ``REGISTRY``, then a name heuristic.

Token counts come from LangChain's real ``usage_metadata`` when a node's output
carries it (the modern ``AIMessage`` field, or a legacy
``response_metadata['token_usage']``); only when no measurable usage is present
does the adapter fall back to a length estimate. Each run is labelled
``tokens=measured`` / ``mixed`` / ``estimated`` accordingly, so no analyzer — and
no dollar figure — mistakes an estimate for a measurement. The model name is read
off the same message, so each node is priced at the model it actually called
rather than one configured default.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


@dataclass
class Usage:
    """Real token usage found in a node update, with the model that billed it."""

    total: int = 0
    prompt: int = 0
    completion: int = 0
    found: bool = False
    model: str | None = None
    mixed_models: bool = False


# Where LangChain providers put the model name on a response. OpenAI writes
# ``model_name``, Anthropic ``model``; LangSmith adds ``ls_model_name``.
_MODEL_KEYS = ("model_name", "model", "ls_model_name")


def _usage_parts(meta: Any) -> tuple[int, int, int] | None:
    """Return ``(total, prompt, completion)`` from a usage mapping, or None."""
    if not isinstance(meta, Mapping):
        return None
    prompt = meta.get("input_tokens")
    if prompt is None:
        prompt = meta.get("prompt_tokens")
    completion = meta.get("output_tokens")
    if completion is None:
        completion = meta.get("completion_tokens")
    total = meta.get("total_tokens")
    if total is None:
        if prompt is None and completion is None:
            return None
        total = int(prompt or 0) + int(completion or 0)
    return int(total), int(prompt or 0), int(completion or 0)


def _model_name(obj: Any) -> str | None:
    """Read the model that produced a message, when the framework reports it."""
    rm = getattr(obj, "response_metadata", None)
    if isinstance(rm, Mapping):
        for key in _MODEL_KEYS:
            value = rm.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _extract_usage(value: Any, seen: set[int] | None = None) -> Usage:
    """Sum real LLM token usage in a node update, if the framework exposes it.

    Walks the update for LangChain message-like objects carrying real usage:
    the modern ``AIMessage.usage_metadata`` (``{"input_tokens", "output_tokens",
    "total_tokens"}``) or a legacy ``response_metadata["token_usage"]``, and reads
    the model name off the same message so cost can be priced per node instead of
    against one global default. ``found`` is False when nothing measurable is
    present, so the caller falls back to the length estimate and the run is
    labelled honestly.

    ``seen`` may be shared across a run's nodes so a message echoed in a later
    node's state (e.g. message-list state without an ``add_messages`` reducer) is
    counted once, at the node that produced it — not re-attributed downstream.
    """
    usage = Usage()
    counted: set[int] = seen if seen is not None else set()  # usage objects already tallied
    local: set[int] = set()  # per-call cycle guard on containers (ids get recycled across calls)
    by_model: dict[str, int] = {}  # tokens per model, to pick the dominant one

    def walk(obj: Any) -> None:
        oid = id(obj)
        if oid in local:
            return
        local.add(oid)

        parts = _usage_parts(getattr(obj, "usage_metadata", None))
        if parts is None:
            rm = getattr(obj, "response_metadata", None)
            if isinstance(rm, Mapping):
                parts = _usage_parts(rm.get("token_usage") or rm.get("usage"))
        if parts is not None:
            # Usage is present, so this update is measured — but only add tokens
            # the first time we see this specific message (echoes count once).
            usage.found = True
            if oid not in counted:
                counted.add(oid)
                total, prompt, completion = parts
                usage.total += total
                usage.prompt += prompt
                usage.completion += completion
                model = _model_name(obj)
                if model:
                    by_model[model] = by_model.get(model, 0) + total
            return  # counted this message; don't recurse into its fields

        if isinstance(obj, Mapping):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(value)
    if by_model:
        # Several models in one node is legal; attribute to the one that spent the
        # most tokens and flag it so the report can say the figure is approximate.
        usage.model = max(by_model, key=lambda name: by_model[name])
        usage.mixed_models = len(by_model) > 1
    return usage


def _observed_tools(value: Any, seen: set[int]) -> list[str]:
    """Names of tools the framework reports this node actually invoked.

    Reads what LangChain already puts in the streamed update — a ``ToolMessage``
    (the tool ran and returned) or an ``AIMessage.tool_calls`` list (the model
    asked for it) — so level-1 works at zero annotation instead of only trusting
    what the caller declared. ``seen`` is shared across a run so a message echoed
    into a later node's state is attributed once, to its producer.
    """
    names: list[str] = []
    local: set[int] = set()

    def add(name: Any) -> None:
        if isinstance(name, str) and name and name not in names:
            names.append(name)

    def walk(obj: Any) -> None:
        oid = id(obj)
        if oid in local:
            return
        local.add(oid)

        if getattr(obj, "type", None) == "tool":       # ToolMessage: it ran
            if oid not in seen:
                seen.add(oid)
                add(getattr(obj, "name", None))
            return

        calls = getattr(obj, "tool_calls", None)        # AIMessage: it asked
        if isinstance(calls, (list, tuple)) and calls:
            if oid not in seen:
                seen.add(oid)
                for call in calls:
                    add(call.get("name") if isinstance(call, Mapping) else getattr(call, "name", None))
            return

        if isinstance(obj, Mapping):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(value)
    return names


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


class _StreamState:
    """Mutable accumulator threaded through a single (sync or async) stream."""

    def __init__(self) -> None:
        self.final_state: Any = None
        self.prev = time.perf_counter()
        self.measured_nodes = 0
        self.billable_nodes = 0  # nodes expected to bill a model (excludes pure tool nodes)
        self.node_count = 0
        self.usage_seen: set[int] = set()  # shared so echoed messages count once
        self.tools_seen: set[int] = set()  # ditto for observed tool calls


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
        # Flipped once the async path is used, so ablation replays the same way.
        self._is_async = False

    # -- attribute passthrough so the wrapper is a drop-in for most callers ---
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial proxy
        return getattr(self._app, item)

    def _tool_calls_for(self, node_id: str, observed: list[str] | None = None) -> list[ToolCallRecord]:
        """Tool records for one node: what was observed, plus what was declared."""
        names: list[str] = list(observed or [])
        for name in self.node_tools.get(node_id, []):
            if name not in names:
                names.append(name)
        records: list[ToolCallRecord] = []
        for name in names:
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

    # -- shared trace-building helpers (used by both sync and async paths) ----
    def _new_run(self, labels: dict[str, str] | None) -> RunTrace:
        return RunTrace(
            run_id=uuid.uuid4().hex[:12],
            graph_name=self.graph_name,
            mode=self.mode,
            labels={"tokens": "estimated", **(labels or {})},
        )

    def _consume(self, run: RunTrace, state: _StreamState, mode: str, chunk: Any) -> None:
        """Fold one streamed ``(mode, chunk)`` pair into the run being built."""
        if mode == "values":
            state.final_state = chunk
            return
        # mode == "updates": {node_id: partial_state}
        now = time.perf_counter()
        latency_ms = int((now - state.prev) * 1000)
        state.prev = now
        for node_id, update in chunk.items():
            text = _stringify(update)
            usage = _extract_usage(update, state.usage_seen)
            tool_calls = self._tool_calls_for(
                node_id, _observed_tools(update, state.tools_seen)
            )
            state.node_count += 1

            # Attribute tokens honestly. A node that reports real usage is
            # measured. A node that reports none but calls an exogenous tool is a
            # retrieval/tool node — it bills no model, so its model cost is $0
            # (never a phantom length estimate). Only a node with no usage *and*
            # no exogenous tool is presumed an unreported model call and estimated.
            is_tool_node = any(tc.exogenous for tc in tool_calls)
            prompt = completion = 0
            if usage.found:
                token_source, tokens = "measured", usage.total
                prompt, completion = usage.prompt, usage.completion
                state.measured_nodes += 1
                state.billable_nodes += 1
            elif is_tool_node:
                token_source, tokens = "none", 0
            else:
                token_source, tokens = "estimated", _estimate_tokens(text)
                state.billable_nodes += 1

            run.add_node(
                NodeRun(
                    node_id=node_id,
                    tool_calls=tool_calls,
                    outcome=Outcome(
                        tokens=tokens,
                        prompt_tokens=prompt,
                        completion_tokens=completion,
                        latency_ms=latency_ms,
                        output_text=text,
                        token_source=token_source,
                        model=usage.model,
                        mixed_models=usage.mixed_models,
                    ),
                )
            )

    def _fail_run(self, run: RunTrace, exc: Exception, record: bool) -> None:
        run.status = RunStatus.ERROR
        run.labels["error"] = str(exc)
        log_event(log, "run failed", stage="adapter.run", graph=self.graph_name, status="error", error=str(exc))
        if record:
            self._store.add(run.finalize())

    def _finish_run(
        self,
        run: RunTrace,
        state: _StreamState,
        labels: dict[str, str] | None,
        record: bool,
        input: Any,
        config: dict | None,
    ) -> RunTrace:
        # Label token provenance honestly: measured (all nodes had real usage),
        # mixed (some did), or the default estimated (none did). Downstream $
        # figures read this so an estimate is never presented as a measurement.
        if "tokens" not in (labels or {}):
            # Pure tool nodes bill no model, so they don't count against "measured".
            if state.billable_nodes and state.measured_nodes == state.billable_nodes:
                run.labels["tokens"] = "measured"
            elif state.measured_nodes:
                run.labels["tokens"] = "mixed"

        run.finalize(self._final_output(state.final_state))
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
        return run

    # -- synchronous execution path ------------------------------------------
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
        """Stream ``graph`` once (sync), building the trace and capturing final state."""
        run = self._new_run(labels)
        state = _StreamState()
        try:
            for mode, chunk in graph.stream(input, config, stream_mode=["updates", "values"]):
                self._consume(run, state, mode, chunk)
        except Exception as exc:  # surface, never swallow (CLAUDE.md)
            self._fail_run(run, exc, record)
            raise
        self._finish_run(run, state, labels, record, input, config)
        return run, state.final_state

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

    # -- asynchronous execution path (for async graphs: nodes are ``async def`` /
    #    driven by ``ainvoke``/``astream``, as in most production LangGraph apps) --
    async def _aexecute(
        self,
        graph: Any,
        input: Any,
        config: dict | None = None,
        *,
        labels: dict[str, str] | None = None,
        record: bool = True,
    ) -> tuple[RunTrace, Any]:
        """Async twin of :meth:`_execute`, using ``graph.astream``."""
        self._is_async = True
        run = self._new_run(labels)
        state = _StreamState()
        try:
            async for mode, chunk in graph.astream(input, config, stream_mode=["updates", "values"]):
                self._consume(run, state, mode, chunk)
        except Exception as exc:  # surface, never swallow (CLAUDE.md)
            self._fail_run(run, exc, record)
            raise
        self._finish_run(run, state, labels, record, input, config)
        return run, state.final_state

    async def arun(
        self,
        graph: Any,
        input: Any,
        config: dict | None = None,
        *,
        labels: dict[str, str] | None = None,
        record: bool = True,
    ) -> RunTrace:
        """Async twin of :meth:`run`."""
        run, _ = await self._aexecute(graph, input, config, labels=labels, record=record)
        return run

    async def areplay_output(self, graph: Any, input: Any, config: dict | None = None) -> str:
        """Async twin of :meth:`replay_output` (used by async ablation)."""
        _, final_state = await self._aexecute(graph, input, config, record=False)
        return self._final_output(final_state)

    def replay_output_blocking(self, graph: Any, input: Any, config: dict | None = None) -> str:
        """Replay (no recording) from a synchronous call site regardless of graph kind.

        Sync ablation drives this. For a sync graph it is just
        :meth:`replay_output`. For an async graph it awaits :meth:`areplay_output`
        on an event loop: ``asyncio.run`` when none is running, otherwise a
        one-shot worker thread with its own loop (so it is safe to call even from
        inside a running loop, e.g. auditing at the end of an async workflow).
        """
        if not self._is_async:
            return self.replay_output(graph, input, config)

        def _run() -> str:
            return asyncio.run(self.areplay_output(graph, input, config))

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _run()  # no loop running here — safe to drive one directly
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result()

    async def ainvoke(self, input: Any, config: dict | None = None, **_: Any) -> Any:
        """Drop-in replacement for ``graph.ainvoke``; records the run as a side effect."""
        _, final_state = await self._aexecute(self._app, input, config)
        return final_state

    def astream(self, input: Any, config: dict | None = None, **kw: Any) -> Any:
        """Passthrough async stream (not recorded); prefer ``ainvoke`` for auditing."""
        return self._app.astream(input, config, **kw)


def instrument_langgraph(app: Any, store: Any, **kwargs: Any) -> InstrumentedApp:
    """Wrap a compiled LangGraph app for observation. See :class:`InstrumentedApp`."""
    return InstrumentedApp(app, store, **kwargs)


__all__ = ["InstrumentedApp", "instrument_langgraph", "resolve_role"]
