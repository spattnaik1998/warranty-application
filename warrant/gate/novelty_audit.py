"""Delegation economics: the a-posteriori half of the admissibility gate.

Even a structurally-exogenous delegation can return redundant information (the
paper's "Wikipedia-on-MMLU *decreases* accuracy" case). After each admitted
worker runs, we measure the posterior shift its observation induced and keep a
running per-delegation-type value statistic. Delegation types whose observations
consistently fail to move the posterior (mean shift < epsilon) are marked
redundant and pruned by the gate on subsequent proposals — delegations must pay
for themselves in information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from warrant.config import get_settings


@dataclass
class _Stat:
    n: int = 0
    total_shift: float = 0.0

    @property
    def mean_shift(self) -> float:
        return self.total_shift / self.n if self.n else 0.0


@dataclass
class DelegationEconomics:
    """Running record of how much each delegation *type* actually pays off."""

    epsilon: float = field(default_factory=lambda: get_settings().novelty_kl_epsilon)
    # Require a few observations before trusting the mean enough to prune.
    min_observations: int = 2
    _stats: dict[str, _Stat] = field(default_factory=dict)

    def record(self, delegation_type: str, posterior_shift: float) -> bool:
        """Record a hop's realized shift; return True if it was redundant."""
        stat = self._stats.setdefault(delegation_type, _Stat())
        stat.n += 1
        stat.total_shift += posterior_shift
        return posterior_shift < self.epsilon

    def is_redundant_type(self, delegation_type: str) -> bool:
        """True if this type has enough history and a mean shift below epsilon."""
        stat = self._stats.get(delegation_type)
        if not stat or stat.n < self.min_observations:
            return False
        return stat.mean_shift < self.epsilon

    def mean_shift(self, delegation_type: str) -> float:
        stat = self._stats.get(delegation_type)
        return stat.mean_shift if stat else 0.0

    def report(self) -> dict[str, dict[str, float]]:
        return {
            t: {"n": s.n, "mean_shift": round(s.mean_shift, 4)}
            for t, s in self._stats.items()
        }
