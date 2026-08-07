"""Unit tests for the individual audit analyzers."""

from __future__ import annotations

from warrant.analysis.cost import dollars, estimate_cost, price_card, price_per_1k
from warrant.analysis.novelty import novelty_audit, unseen_token_fraction
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


def test_unseen_token_fraction_bounds() -> None:
    assert unseen_token_fraction("a b c", "a b c") == 0.0        # fully redundant
    assert unseen_token_fraction("x y z", "a b c") == 1.0        # fully novel
    assert unseen_token_fraction("", "anything") == 0.0
    assert 0.0 < unseen_token_fraction("a b q", "a b c") < 1.0   # partial
    # Asymmetric on purpose: a long context must not dilute a novel output.
    assert unseen_token_fraction("x y z", "a b c " * 500) == 1.0


def test_novelty_audit_marks_restatement_low() -> None:
    nov = novelty_audit([_run()])
    assert nov["echo"].mean_novelty == 0.0                  # echo restates fetch
    assert nov["fetch"].mean_novelty > 0.0                  # first node is all-new


def test_price_card_splits_input_and_output() -> None:
    card, fallback = price_card("gpt-4o")
    assert (card.input_per_1k, card.output_per_1k) == (0.0025, 0.0100)
    assert fallback is False
    # Output is always the dearer side, so the blended rate sits between them.
    assert card.input_per_1k < card.blended_per_1k < card.output_per_1k

    unknown, fallback = price_card("some-local-llm")
    assert fallback is True                                  # named, not hidden
    assert unknown.output_per_1k > 0


def test_env_overrides_each_side(monkeypatch) -> None:
    monkeypatch.setenv("WARRANT_PRICE_GPT_4O_IN", "0.001")
    card, _ = price_card("gpt-4o")
    assert card.input_per_1k == 0.001
    assert card.output_per_1k == 0.0100                      # untouched side stands


def test_split_tokens_bill_at_their_own_rates() -> None:
    """A prompt-heavy node must not be charged the output rate for its context."""
    split = estimate_cost(1000, "gpt-4o", prompt_tokens=900, completion_tokens=100)
    assert split.dollars_per_run == 900 / 1000 * 0.0025 + 100 / 1000 * 0.0100
    assert split.priced_blended is False

    blended = estimate_cost(1000, "gpt-4o")                  # only a total known
    assert blended.priced_blended is True
    assert blended.dollars_per_run > split.dollars_per_run   # blending overcharges


def test_monthly_cost_requires_a_declared_volume() -> None:
    measured = estimate_cost(500, "gpt-4o-mini")
    assert measured.dollars_per_month is None                # never invented
    assert measured.dollars_per_1k_runs == measured.dollars_per_run * 1000

    declared = estimate_cost(500, "gpt-4o-mini", runs_per_month=10_000)
    assert declared.dollars_per_month == declared.dollars_per_run * 10_000


def test_dollars_monotonic_in_tokens() -> None:
    assert dollars(2000, "gpt-4o") > dollars(1000, "gpt-4o")
    assert price_per_1k("gpt-4o-mini") < price_per_1k("gpt-4o")
