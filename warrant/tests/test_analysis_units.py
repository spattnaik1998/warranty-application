"""Unit tests for the individual audit analyzers."""

from __future__ import annotations

from warrant.analysis.cost import dollars, estimate_cost, price_per_1k
from warrant.analysis.novelty import jaccard_distance, novelty_audit
from warrant.analysis.structural import structural_audit
from warrant.schemas.belief import AdmissibilityClass
from warrant.tools.registry import ToolRole
from warrant.trace.contract import NodeRun, Outcome, RunTrace, ToolCallRecord


def _run(run_id: str = "r") -> RunTrace:
    run = RunTrace(run_id=run_id, graph_name="g")
    run.add_node(
        NodeRun(
            node_id="fetch",
            tool_calls=[ToolCallRecord(name="arxiv", role=ToolRole.INJECTOR)],
            outcome=Outcome(tokens=100, output_text="alpha beta gamma delta"),
        )
    )
    run.add_node(
        NodeRun(
            node_id="echo",
            outcome=Outcome(tokens=40, output_text="alpha beta gamma delta"),  # pure restatement
        )
    )
    return run.finalize("alpha beta gamma delta")


def test_structural_flags_no_tool_node_as_reorganizer() -> None:
    struct = structural_audit([_run()])
    assert struct["fetch"].admissibility is AdmissibilityClass.INJECTOR
    assert struct["echo"].admissibility is AdmissibilityClass.REORGANIZER
    assert struct["echo"].is_reorganizer is True


def test_jaccard_distance_bounds() -> None:
    assert jaccard_distance("a b c", "a b c") == 0.0        # fully redundant
    assert jaccard_distance("x y z", "a b c") == 1.0        # fully novel
    assert jaccard_distance("", "anything") == 0.0
    assert 0.0 < jaccard_distance("a b q", "a b c") < 1.0   # partial


def test_novelty_audit_marks_restatement_low() -> None:
    nov = novelty_audit([_run()])
    assert nov["echo"].mean_novelty == 0.0                  # echo restates fetch
    assert nov["fetch"].mean_novelty > 0.0                  # first node is all-new


def test_cost_math() -> None:
    assert price_per_1k("gpt-4o-mini") == 0.0004
    assert price_per_1k("unknown-model") == 0.005           # fallback
    assert dollars(1000, "gpt-4o-mini") == 0.0004
    est = estimate_cost(500, 3, model="gpt-4o-mini", runs_per_month=10_000)
    assert est.dollars_per_run == 500 / 1000 * 0.0004
    assert est.dollars_per_month == est.dollars_per_run * 10_000


def test_novelty_reuses_dollars_monotonic() -> None:
    # more tokens -> strictly more dollars
    assert dollars(2000, "gpt-4o") > dollars(1000, "gpt-4o")
