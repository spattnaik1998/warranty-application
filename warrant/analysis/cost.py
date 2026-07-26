"""Token / latency → dollar attribution.

Turns each node's token usage into a monthly cost so a redundant delegation
becomes a CFO-legible number ("collapsing reviewer saves ~$Y/mo"). Prices are a
small, overridable table keyed by model name; unknown models fall back to a
conservative default so the audit never silently reports $0 for real spend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# USD per 1K tokens (blended prompt+completion), overridable via WARRANT_PRICE_<MODEL>.
_DEFAULT_PRICES: dict[str, float] = {
    "gpt-4o": 0.0075,
    "gpt-4o-mini": 0.0004,
    "claude-opus-4-8": 0.030,
    "claude-sonnet-5": 0.006,
    "claude-haiku-4-5": 0.0016,
}
_FALLBACK_PRICE_PER_1K = 0.005


def price_per_1k(model: str | None) -> float:
    """Blended price per 1K tokens for a model (env override > table > fallback)."""
    if not model:
        return _FALLBACK_PRICE_PER_1K
    env = os.getenv("WARRANT_PRICE_" + model.upper().replace("-", "_").replace(".", "_"))
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _DEFAULT_PRICES.get(model, _FALLBACK_PRICE_PER_1K)


@dataclass(frozen=True)
class CostEstimate:
    """Dollar cost of a set of node runs, per-run and projected monthly."""

    tokens: int
    dollars_per_run: float
    runs_per_month: int
    dollars_per_month: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tokens": self.tokens,
            "dollars_per_run": round(self.dollars_per_run, 6),
            "runs_per_month": self.runs_per_month,
            "dollars_per_month": round(self.dollars_per_month, 2),
        }


def dollars(tokens: int, model: str | None = None) -> float:
    """Cost in USD of ``tokens`` tokens at ``model``'s blended rate."""
    return tokens / 1000.0 * price_per_1k(model)


def estimate_cost(
    tokens_per_run: int,
    n_runs_observed: int,
    model: str | None = None,
    runs_per_month: int = 30_000,
) -> CostEstimate:
    """Project the monthly dollar cost of a delegation seen ``n_runs_observed`` times.

    ``tokens_per_run`` is the mean tokens the delegation used per run; the
    projection assumes a nominal ``runs_per_month`` production volume (overridable
    so the headline number reflects the customer's real traffic).
    """
    per_run = dollars(tokens_per_run, model)
    return CostEstimate(
        tokens=tokens_per_run,
        dollars_per_run=per_run,
        runs_per_month=runs_per_month,
        dollars_per_month=per_run * runs_per_month,
    )


__all__ = ["CostEstimate", "estimate_cost", "dollars", "price_per_1k"]
