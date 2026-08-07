"""Trace Contract schema round-trips and store persistence."""

from __future__ import annotations

from warrant.schemas.belief import AdmissibilityClass, Posterior
from warrant.tools.registry import ToolRole
from warrant.trace.contract import (
    DecisionPoint,
    NodeRun,
    Outcome,
    RunStatus,
    RunTrace,
    ToolCallRecord,
    content_digest,
)
from warrant.trace.store import SQLiteTraceStore, TraceStore


def _sample_run() -> RunTrace:
    run = RunTrace(run_id="r1", graph_name="demo")
    run.add_node(
        NodeRun(
            node_id="retriever",
            instruction="fetch papers",
            tool_calls=[
                ToolCallRecord(name="arxiv", role=ToolRole.INJECTOR, tokens=50, latency_ms=200)
            ],
            outcome=Outcome(tokens=120, latency_ms=300, output_text="found 3 papers"),
        )
    )
    run.add_node(
        NodeRun(
            node_id="compose",
            instruction="reorganize context",
            tool_calls=[ToolCallRecord(name="format", role=ToolRole.TRANSFORM)],
            decision=DecisionPoint(
                key="topic", options=["a", "b"], posterior=Posterior(probs={"a": 0.7, "b": 0.3})
            ),
            outcome=Outcome(tokens=80, latency_ms=150, output_text="a briefing"),
        )
    )
    return run.finalize(final_output="a briefing")


def test_node_exogeneity_flags() -> None:
    run = _sample_run()
    assert run.node("retriever").has_exogenous_tool is True
    assert run.node("compose").has_exogenous_tool is False


def test_outcome_digest_autofilled() -> None:
    run = _sample_run()
    assert run.node("retriever").outcome.output_digest == content_digest("found 3 papers")
    assert run.final_digest == content_digest("a briefing")


def test_totals() -> None:
    run = _sample_run()
    assert run.total_tokens == 200
    assert run.total_latency_ms == 450


def test_contract_json_round_trip() -> None:
    run = _sample_run()
    run.node("compose").admissibility = AdmissibilityClass.REORGANIZER
    payload = run.model_dump_json()
    back = RunTrace.model_validate_json(payload)
    assert back == run
    assert back.node("compose").admissibility is AdmissibilityClass.REORGANIZER
    assert back.node("compose").decision.posterior.top_label() == "a"


def test_inmemory_store() -> None:
    store = TraceStore()
    store.add(_sample_run())
    assert len(store) == 1
    assert store.get("r1").graph_name == "demo"
    store.clear()
    assert len(store) == 0


def test_sqlite_store_persists_and_reloads(tmp_path) -> None:
    db = tmp_path / "traces.db"
    store = SQLiteTraceStore(db)
    store.add(_sample_run())
    store.close()

    reopened = SQLiteTraceStore(db)
    assert len(reopened) == 1
    reloaded = reopened.get("r1")
    assert reloaded is not None
    assert reloaded.status is RunStatus.OK
    assert reloaded.node("compose").decision.posterior.top_label() == "a"
    reopened.close()


def test_sqlite_clear_deletes_from_disk_too(tmp_path) -> None:
    """A cleared store must stay cleared: `session()` scopes runs, and a trace
    that reappeared on the next open would silently widen the next audit."""
    db = tmp_path / "traces.db"
    store = SQLiteTraceStore(db)
    store.add(_sample_run())
    store.clear()
    assert len(store) == 0
    assert store.load() == []            # nothing left behind in the table
    store.close()

    assert len(SQLiteTraceStore(db)) == 0  # and nothing resurrects on reopen
