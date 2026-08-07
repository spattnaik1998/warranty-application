"""Token → dollar attribution.

Turns each node's token usage into a dollar figure so a redundant delegation
becomes a CFO-legible number ("collapsing reviewer saves ~$Y per 1,000 runs").
Prices are a small, overridable table of input/output rates keyed by model name;
unknown models fall back to a conservative default so the audit never silently
reports $0 for real spend.

Two things this module deliberately does *not* do: invent a traffic volume (the
caller declares ``runs_per_month`` or gets a per-1,000-runs figure), and hide
which rate it used (``CostEstimate.priced_blended`` and ``.fallback_price`` say
so, and the report surfaces both).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# USD per 1K tokens, split input/output. Anthropic rates from the published
# price list; OpenAI rates from theirs. Override either side per model with
# WARRANT_PRICE_<MODEL>_IN / _OUT, or both at once with WARRANT_PRICE_<MODEL>.
_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (0.0025, 0.0100),
    "gpt-4o-mini": (0.00015, 0.0006),
    # Anthropic
    "claude-fable-5": (0.010, 0.050),
    "claude-opus-5": (0.005, 0.025),
    "claude-opus-4-8": (0.005, 0.025),
    "claude-sonnet-5": (0.003, 0.015),
    "claude-haiku-4-5": (0.001, 0.005),
}
_FALLBACK_PRICE = (0.003, 0.015)

# When only a token *total* is known, input and output must be blended into one
# rate. Agent nodes carry long context and emit short answers, so the split is
# assumed prompt-heavy. This is an assumption, and the report says so whenever a
# node is priced this way.
_BLENDED_INPUT_SHARE = 0.75


def _env_price(model: str, suffix: str) -> float | None:
    """Read WARRANT_PRICE_<MODEL>[_IN|_OUT], or None if unset/unparseable."""
    key = "WARRANT_PRICE_" + model.upper().replace("-", "_").replace(".", "_") + suffix
    raw = os.getenv(key)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class PriceCard:
    """Per-1K-token input and output rates for one model."""

    input_per_1k: float
    output_per_1k: float

    @property
    def blended_per_1k(self) -> float:
        """One rate for when the input/output split is unknown (see module docs)."""
        share = _BLENDED_INPUT_SHARE
        return self.input_per_1k * share + self.output_per_1k * (1 - share)


def price_card(model: str | None) -> tuple[PriceCard, bool]:
    """Return ``(card, is_fallback)`` for a model: env override > table > default.

    ``is_fallback`` is True when nothing knew this model's real rates, so the
    report can name it rather than quietly billing a made-up number.
    """
    if not model:
        return PriceCard(*_FALLBACK_PRICE), True

    listed = _DEFAULT_PRICES.get(model)
    both = _env_price(model, "")           # legacy single-rate override
    if listed is not None:
        base, is_fallback = listed, False
    elif both is not None:
        base, is_fallback = (both, both), False
    else:
        base, is_fallback = _FALLBACK_PRICE, True

    side_in = _env_price(model, "_IN")
    side_out = _env_price(model, "_OUT")
    if side_in is not None or side_out is not None:
        is_fallback = False
    return (
        PriceCard(
            input_per_1k=base[0] if side_in is None else side_in,
            output_per_1k=base[1] if side_out is None else side_out,
        ),
        is_fallback,
    )


def price_per_1k(model: str | None) -> float:
    """Blended price per 1K tokens for a model (the total-tokens-only path)."""
    return price_card(model)[0].blended_per_1k


@dataclass(frozen=True)
class CostEstimate:
    """Dollar cost of one node's runs, per run and per 1,000 runs.

    ``dollars_per_month`` is populated only when the caller declared a traffic
    volume; ``None`` means nobody told Warrant how often this graph runs, and it
    refuses to guess.
    """

    tokens: int
    model: str | None
    dollars_per_run: float
    dollars_per_1k_runs: float
    runs_per_month: int | None = None
    dollars_per_month: float | None = None
    priced_blended: bool = False   # input/output split unknown; blended rate used
    fallback_price: bool = False   # model not in the price table

    def as_dict(self) -> dict[str, object]:
        return {
            "tokens": self.tokens,
            "model": self.model,
            "dollars_per_run": round(self.dollars_per_run, 6),
            "dollars_per_1k_runs": round(self.dollars_per_1k_runs, 4),
            "runs_per_month": self.runs_per_month,
            "dollars_per_month": (
                None if self.dollars_per_month is None else round(self.dollars_per_month, 2)
            ),
            "priced_blended": self.priced_blended,
            "fallback_price": self.fallback_price,
        }


def dollars(tokens: int, model: str | None = None) -> float:
    """Cost in USD of ``tokens`` tokens at ``model``'s blended rate."""
    return tokens / 1000.0 * price_per_1k(model)


def estimate_cost(
    tokens_per_run: int,
    model: str | None = None,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    runs_per_month: int | None = None,
) -> CostEstimate:
    """Price one delegation's mean per-run token usage.

    When ``prompt_tokens``/``completion_tokens`` are known they are billed at
    their own rates; otherwise ``tokens_per_run`` is billed at the blended rate
    and the estimate is flagged ``priced_blended``. ``runs_per_month`` is the
    caller's declared production volume — omit it and only the measured
    per-run and per-1,000-run figures are returned.
    """
    card, is_fallback = price_card(model)
    split_known = (prompt_tokens + completion_tokens) > 0
    if split_known:
        per_run = (
            prompt_tokens / 1000.0 * card.input_per_1k
            + completion_tokens / 1000.0 * card.output_per_1k
        )
    else:
        per_run = tokens_per_run / 1000.0 * card.blended_per_1k
    return CostEstimate(
        tokens=tokens_per_run,
        model=model,
        dollars_per_run=per_run,
        dollars_per_1k_runs=per_run * 1000.0,
        runs_per_month=runs_per_month,
        dollars_per_month=None if runs_per_month is None else per_run * runs_per_month,
        priced_blended=not split_known and tokens_per_run > 0,
        fallback_price=is_fallback and tokens_per_run > 0,
    )


__all__ = ["CostEstimate", "PriceCard", "estimate_cost", "dollars", "price_per_1k", "price_card"]
