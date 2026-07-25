"""Scoring helpers for the ledger: accuracy, relay degradation, distortion."""

from __future__ import annotations

import hashlib

from warrant.belief.state import BeliefState
from warrant.pipeline.steps import DecisionKey
from warrant.schemas.belief import Posterior


def accuracy(beliefs: BeliefState, dks: list[DecisionKey]) -> float:
    """Fraction of decision keys whose top label matches ground truth."""
    if not dks:
        return 0.0
    correct = 0
    for dk in dks:
        post = beliefs.posterior(dk.key)
        if post and post.top_label() == dk.correct:
            correct += 1
    return correct / len(dks)


def _survives(key: str, hops: int, per_hop_loss: float = 0.22) -> bool:
    """Deterministic model of whether a fact survives `hops` prose relays.

    Survival probability decays linearly with relay depth; realised
    deterministically per key so runs are reproducible. This is the offline
    stand-in for the paper's measured prose-relay information loss.
    """
    survival_p = max(0.0, 1.0 - per_hop_loss * hops)
    draw = (int(hashlib.sha1(f"{key}|{hops}".encode()).hexdigest(), 16) % 1000) / 1000.0
    return draw < survival_p


def relayed_context(
    dk: DecisionKey,
    source_text: str,
    hops: int,
    interface: str = "prose",
) -> str:
    """Build the context a decision point sees after `hops` relay stages.

    * ``interface == 'posterior'``: the structured posterior is carried forward
      losslessly, so the evidence keyword is always preserved (no relay loss).
    * ``interface == 'prose'``: each relay may drop the exact figure; once
      dropped, the decision point can no longer ground the claim.
    """
    keyword = next((kw for kws in dk.hints.values() for kw in kws if kw), "")
    # Locate the source sentence mentioning the keyword.
    sentence = ""
    for chunk in source_text.replace("\n", " ").split(". "):
        if keyword and keyword.lower() in chunk.lower():
            sentence = chunk.strip()
            break
    if not sentence:
        sentence = source_text[:200]

    if interface == "posterior" or hops == 0 or _survives(dk.key, hops):
        return sentence
    # Fact dropped by the prose relay: keep a generic paraphrase without the figure.
    return "The paper discusses results on a benchmark relevant to the topic."


def distortion_to_centralized(
    beliefs: BeliefState,
    centralized: dict[str, Posterior],
    dks: list[DecisionKey],
) -> float:
    """Total Brier distortion of a run's posteriors vs the centralized baseline."""
    total = 0.0
    for dk in dks:
        base = centralized.get(dk.key)
        post = beliefs.posterior(dk.key)
        if base and post:
            total += post.brier_to(base)
    return total


def mean_kl_to_centralized(
    beliefs: BeliefState,
    centralized: dict[str, Posterior],
    dks: list[DecisionKey],
) -> float:
    """Mean KL(centralized || run) across keys — the per-run relay distortion."""
    kls = []
    for dk in dks:
        base = centralized.get(dk.key)
        post = beliefs.posterior(dk.key)
        if base and post:
            kls.append(base.kl_to(post))
    return sum(kls) / len(kls) if kls else 0.0
