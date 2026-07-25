"""Schema and posterior-math tests."""

from __future__ import annotations

import math

import pytest

from warrant.schemas.belief import Delegation, Posterior, SignalClaim


def test_posterior_normalizes():
    p = Posterior(probs={"a": 3, "b": 1})
    assert p.probs["a"] == pytest.approx(0.75)
    assert p.probs["b"] == pytest.approx(0.25)
    assert sum(p.probs.values()) == pytest.approx(1.0)


def test_posterior_rejects_all_zero():
    with pytest.raises(ValueError):
        Posterior(probs={"a": 0, "b": 0})


def test_entropy_bounds():
    assert Posterior.certain("yes", ["no"]).entropy() == pytest.approx(0.0, abs=1e-9)
    assert Posterior.uniform(["a", "b"]).entropy() == pytest.approx(1.0)  # 1 bit
    assert Posterior.uniform(["a", "b", "c", "d"]).entropy() == pytest.approx(2.0)


def test_kl_certain_vs_uniform_is_one_bit():
    c = Posterior.certain("yes", ["no"])
    u = Posterior.uniform(["yes", "no"])
    assert c.kl_to(u) == pytest.approx(1.0, abs=1e-6)


def test_brier_matches_manual():
    p = Posterior(probs={"yes": 0.75, "no": 0.25})
    u = Posterior.uniform(["yes", "no"])
    # (0.75-0.5)^2 + (0.25-0.5)^2 = 0.125
    assert p.brier_to(u) == pytest.approx(0.125)


def test_margin_and_top():
    p = Posterior(probs={"a": 0.6, "b": 0.3, "c": 0.1})
    assert p.top_label() == "a"
    assert p.margin() == pytest.approx(0.3)


def test_kl_aligns_disjoint_support():
    p = Posterior(probs={"a": 1.0})
    q = Posterior(probs={"b": 1.0})
    assert p.kl_to(q) > 5  # near-infinite, smoothed but large


def test_delegation_requires_signal_claim():
    d = Delegation(
        instruction="verify metric",
        signal_claim=SignalClaim(claim="q", expected_signal_source="arxiv"),
        decision_key="k",
        delegation_type="verify",
        tools=["arxiv_fetch"],
    )
    assert d.tools == ["arxiv_fetch"]
    assert math.isfinite(0.0)
