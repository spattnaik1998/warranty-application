"""Distortion-metric and belief-state tests."""

from __future__ import annotations

import pytest

from warrant.belief.distortion import pearson, serial_chain_loss
from warrant.belief.state import BeliefState
from warrant.schemas.belief import Posterior


def test_serial_chain_loss_zero_for_identical():
    p = Posterior(probs={"a": 0.7, "b": 0.3})
    assert serial_chain_loss([p, p, p]) == pytest.approx(0.0, abs=1e-9)


def test_serial_chain_loss_additive_and_nonnegative():
    a = Posterior(probs={"x": 0.9, "y": 0.1})
    b = Posterior(probs={"x": 0.7, "y": 0.3})
    c = Posterior(probs={"x": 0.5, "y": 0.5})
    total = serial_chain_loss([a, b, c])
    assert total == pytest.approx(a.kl_to(b) + b.kl_to(c))
    assert total > 0


def test_pearson_perfect_correlation():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
    assert pearson([1, 1], [1, 1]) == 0.0  # degenerate


def test_belief_update_shift_from_uniform_is_large():
    bs = BeliefState()
    confident = Posterior(probs={"supported": 0.95, "unsupported": 0.05})
    shift = bs.update("m", confident, claim="c")
    assert shift > 0.5
    assert bs.posterior("m").top_label() == "supported"


def test_belief_redundant_update_has_zero_shift():
    bs = BeliefState()
    p = Posterior(probs={"a": 0.8, "b": 0.2})
    bs.update("k", p)
    shift2 = bs.update("k", p)
    assert shift2 == pytest.approx(0.0, abs=1e-9)
