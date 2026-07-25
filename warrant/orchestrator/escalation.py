"""Risk-triggered human escalation (the paper's Theorem 10).

Escalate a decision to human review iff the automated posterior risk exceeds the
cost of review — R_a(H) > R_h(H) — never by fixed step counts or "the run was
long". R_a is the expected cost of acting on the current terminal belief; for a
0/1 loss it is wrong_cost * (1 - top_probability).
"""

from __future__ import annotations

from warrant.belief.state import BeliefState
from warrant.config import Settings, get_settings
from warrant.pipeline.steps import DecisionKey
from warrant.schemas.belief import Posterior
from warrant.schemas.tasks import Claim


def automated_risk(posterior: Posterior, wrong_cost: float) -> float:
    """Expected cost of committing to the top option under 0/1 loss."""
    return wrong_cost * (1.0 - posterior.top_prob())


def should_escalate(posterior: Posterior, settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return automated_risk(posterior, s.wrong_cost) > s.review_cost


def flag_for_review(
    beliefs: BeliefState,
    dks: list[DecisionKey],
    settings: Settings | None = None,
) -> list[Claim]:
    """Return claims whose terminal posterior risk warrants human review."""
    s = settings or get_settings()
    flagged: list[Claim] = []
    for dk in dks:
        post = beliefs.posterior(dk.key)
        if post is None:
            continue
        if should_escalate(post, s):
            msg = beliefs.get(dk.key)
            flagged.append(Claim(
                text=f"{dk.claim} (risk={automated_risk(post, s.wrong_cost):.2f} "
                     f"> review cost {s.review_cost:.2f})",
                evidence_refs=msg.evidence_refs if msg else [],
                confidence=post.top_prob(),
            ))
    return flagged
