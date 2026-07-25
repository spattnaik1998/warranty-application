"""The orchestrator's belief state over load-bearing decision keys.

The belief state is the shared context the Reliability-Limits paper reasons
about: a set of posteriors the terminal decision depends on. Every worker
observation updates it, and the *size of that update* (posterior shift) is what
the novelty audit uses to decide whether a delegation earned its warrant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from warrant.schemas.belief import BeliefMessage, EvidenceRef, Posterior


@dataclass
class BeliefState:
    """Mutable map of decision_key -> current BeliefMessage."""

    beliefs: dict[str, BeliefMessage] = field(default_factory=dict)
    # Full history of messages per key, for auditing and serial-chain loss.
    history: dict[str, list[BeliefMessage]] = field(default_factory=dict)

    def get(self, key: str) -> BeliefMessage | None:
        return self.beliefs.get(key)

    def posterior(self, key: str) -> Posterior | None:
        msg = self.beliefs.get(key)
        return msg.posterior if msg else None

    def update(
        self,
        key: str,
        posterior: Posterior,
        *,
        claim: str = "",
        evidence_refs: list[EvidenceRef] | None = None,
        summary: str = "",
    ) -> float:
        """Incorporate a new posterior for `key`; return the posterior shift (bits).

        The shift is KL(new || prior), where the prior is the previous belief for
        this key or a uniform distribution over the same support if this is the
        first observation. A large shift means the observation carried genuinely
        new decision-relevant information.
        """
        prior = self.beliefs.get(key)
        prior_post = prior.posterior if prior else Posterior.uniform(posterior.support)
        shift = posterior.kl_to(prior_post)
        msg = BeliefMessage(
            key=key,
            claim=claim or (prior.claim if prior else key),
            posterior=posterior,
            evidence_refs=evidence_refs or [],
            confidence=posterior.top_prob(),
            summary=summary,
        )
        self.beliefs[key] = msg
        self.history.setdefault(key, []).append(msg)
        return shift

    def entropy(self, key: str) -> float:
        post = self.posterior(key)
        return post.entropy() if post else float("inf")

    def keys(self) -> list[str]:
        return list(self.beliefs)

    def snapshot(self) -> dict[str, Posterior]:
        return {k: v.posterior for k, v in self.beliefs.items()}
