"""Posterior-distortion metrics — the empirical core of the reliability thesis.

Communication loss equals posterior distortion (paper's Theorem 8). We provide:

* :func:`serial_chain_loss` — additive KL along a relay chain (the telephone game).
* :func:`distortion_vs_centralized` — how far a hop's posterior sits from the
  centralized (all-evidence) baseline; the "ground truth" upper bound.
* :func:`communication_loss` — aggregate distortion for a run.
* :func:`pearson` — correlation, used to reproduce the KL↔accuracy-drop result.
"""

from __future__ import annotations

import math

from warrant.schemas.belief import Posterior


def serial_chain_loss(chain: list[Posterior]) -> float:
    """Additive posterior distortion along a relay chain, in bits.

    Sum of KL(stage_i || stage_{i+1}) over consecutive stages. For a lossless
    relay this is 0; each garbling hop adds a non-negative term.
    """
    if len(chain) < 2:
        return 0.0
    return sum(chain[i].kl_to(chain[i + 1]) for i in range(len(chain) - 1))


def distortion_vs_centralized(hop: Posterior, centralized: Posterior) -> float:
    """Brier distortion ||hop - centralized||^2 against the all-evidence baseline."""
    return hop.brier_to(centralized)


def communication_loss(
    hop_posteriors: dict[str, Posterior],
    centralized_posteriors: dict[str, Posterior],
) -> float:
    """Total distortion vs centralized across all shared decision keys."""
    total = 0.0
    for key, centralized in centralized_posteriors.items():
        hop = hop_posteriors.get(key)
        if hop is not None:
            total += distortion_vs_centralized(hop, centralized)
    return total


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient; returns 0.0 for degenerate input."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else 0.0
