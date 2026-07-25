"""Admissibility gate and delegation-economics tests."""

from __future__ import annotations

import warrant.tools  # noqa: F401  (registers tools)
from warrant.gate import DelegationEconomics
from warrant.gate.admissibility import classify_delegation, gate_decision
from warrant.schemas.belief import AdmissibilityClass, Delegation, SignalClaim
from warrant.tools import REGISTRY


def _delegation(tools, dtype="t"):
    return Delegation(
        instruction="do",
        signal_claim=SignalClaim(claim="q"),
        decision_key="k",
        delegation_type=dtype,
        tools=tools,
    )


def test_injector_is_admitted():
    d = _delegation(["arxiv_fetch"])
    assert classify_delegation(d, REGISTRY) is AdmissibilityClass.INJECTOR
    assert gate_decision(d, REGISTRY).admit is True


def test_validator_is_admitted():
    d = _delegation(["factcheck_metric"])
    assert classify_delegation(d, REGISTRY) is AdmissibilityClass.VALIDATOR
    assert gate_decision(d, REGISTRY).admit is True


def test_reorganizer_is_rejected():
    d = _delegation(["draft_text"])
    assert classify_delegation(d, REGISTRY) is AdmissibilityClass.REORGANIZER
    decision = gate_decision(d, REGISTRY)
    assert decision.admit is False
    assert "REORGANIZER" in decision.reason


def test_empty_toolset_is_reorganizer():
    d = _delegation([])
    assert classify_delegation(d, REGISTRY) is AdmissibilityClass.REORGANIZER


def test_injector_beats_validator_and_transform():
    d = _delegation(["draft_text", "factcheck_metric", "arxiv_fetch"])
    assert classify_delegation(d, REGISTRY) is AdmissibilityClass.INJECTOR


def test_economics_prunes_redundant_type():
    econ = DelegationEconomics(epsilon=0.1, min_observations=2)
    # Two near-zero shifts flag the type as redundant.
    econ.record("noisy_fetch", 0.0)
    econ.record("noisy_fetch", 0.01)
    assert econ.is_redundant_type("noisy_fetch") is True

    d = _delegation(["arxiv_fetch"], dtype="noisy_fetch")
    decision = gate_decision(d, REGISTRY, econ)
    assert decision.admit is False
    assert decision.redundant is True


def test_economics_keeps_valuable_type():
    econ = DelegationEconomics(epsilon=0.05, min_observations=2)
    econ.record("good_fetch", 0.9)
    econ.record("good_fetch", 0.8)
    assert econ.is_redundant_type("good_fetch") is False
    d = _delegation(["arxiv_fetch"], dtype="good_fetch")
    assert gate_decision(d, REGISTRY, econ).admit is True
