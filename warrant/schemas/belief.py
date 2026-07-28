"""Belief- and delegation-related schemas.

These are the load-bearing contracts of the whole system:

* :class:`Posterior` — a probability distribution over discrete options, with
  the information-theoretic operations the reliability thesis needs
  (entropy, margin, KL, Brier). Inter-agent messages carry these instead of
  prose, implementing the paper's "structured posterior" interface that
  degrades ~2.8 pts/stage versus ~8.5 for free-form relay.
* :class:`SignalClaim` / :class:`Delegation` — the four-tuple Φ plus a
  declaration of the *new exogenous signal* a worker promises to inject.
* :class:`Observation` — the structured result a worker returns.
* :class:`GateDecision` — the admissibility gate's verdict.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

# Smoothing constant so KL / log terms stay finite on zero-support options.
_EPS = 1e-9


class Posterior(BaseModel):
    """A probability distribution over a discrete set of labelled options.

    Probabilities are normalized on construction. All information-theoretic
    quantities are reported in **bits** (log base 2) for interpretability.
    """

    probs: dict[str, float] = Field(..., description="label -> probability")

    model_config = {"frozen": True}

    @field_validator("probs")
    @classmethod
    def _check_nonempty_nonneg(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("posterior must have at least one option")
        for label, p in v.items():
            if p < 0 or math.isnan(p):
                raise ValueError(f"probability for {label!r} must be >= 0, got {p}")
        return v

    @model_validator(mode="after")
    def _normalize(self) -> "Posterior":
        total = sum(self.probs.values())
        if total <= 0:
            raise ValueError("posterior probabilities sum to zero")
        # Rebuild normalized dict (model is frozen; use object.__setattr__).
        normalized = {k: v / total for k, v in self.probs.items()}
        object.__setattr__(self, "probs", normalized)
        return self

    # --- constructors -----------------------------------------------------

    @classmethod
    def uniform(cls, labels: list[str]) -> "Posterior":
        p = 1.0 / len(labels)
        return cls(probs={label: p for label in labels})

    @classmethod
    def certain(cls, label: str, others: list[str] | None = None) -> "Posterior":
        """A near-degenerate posterior placing all mass on `label`."""
        probs = {label: 1.0}
        for o in others or []:
            probs.setdefault(o, 0.0)
        return cls(probs=probs)

    # --- summaries --------------------------------------------------------

    @property
    def support(self) -> list[str]:
        return list(self.probs.keys())

    def top_label(self) -> str:
        return max(self.probs.items(), key=lambda kv: kv[1])[0]

    def top_prob(self) -> float:
        return max(self.probs.values())

    def entropy(self) -> float:
        """Shannon entropy in bits. 0 = certain, log2(n) = uniform."""
        return -sum(p * math.log2(p) for p in self.probs.values() if p > 0)

    def margin(self) -> float:
        """Top-1 minus top-2 probability. High margin = confident decision."""
        vals = sorted(self.probs.values(), reverse=True)
        return vals[0] - (vals[1] if len(vals) > 1 else 0.0)

    # --- pairwise divergences (support is aligned by union) ---------------

    def _aligned(self, other: "Posterior") -> tuple[list[float], list[float]]:
        labels = set(self.probs) | set(other.probs)
        p = [self.probs.get(label, 0.0) for label in labels]
        q = [other.probs.get(label, 0.0) for label in labels]
        return p, q

    def kl_to(self, other: "Posterior") -> float:
        """KL(self || other) in bits, with epsilon smoothing for stability."""
        p, q = self._aligned(other)
        return sum(
            pi * math.log2((pi + _EPS) / (qi + _EPS))
            for pi, qi in zip(p, q)
            if pi > 0
        )

    def brier_to(self, other: "Posterior") -> float:
        """Squared L2 distance ||self - other||^2 between the two vectors.

        This is exactly the communication loss the paper derives under the
        Brier scoring rule (expected squared posterior error).
        """
        p, q = self._aligned(other)
        return sum((pi - qi) ** 2 for pi, qi in zip(p, q))


class AdmissibilityClass(str, Enum):
    """The three delegation classes from the paper's Corollary 9."""

    INJECTOR = "INJECTOR"        # brings a new exogenous signal -> admit
    VALIDATOR = "VALIDATOR"      # non-redundant external/executable check -> admit
    REORGANIZER = "REORGANIZER"  # only re-reads shared context -> reject/collapse


class EvidenceRef(BaseModel):
    """A pointer to a concrete exogenous observation backing a claim."""

    source_type: str = Field(..., description="e.g. 'arxiv', 'pdf', 'web', 'youtube'")
    source_id: str = Field(..., description="e.g. arXiv id, URL, video id")
    locator: str | None = Field(None, description="page/span/section within the source")
    snippet: str | None = Field(None, description="short quoted evidence")

    def short(self) -> str:
        loc = f"#{self.locator}" if self.locator else ""
        return f"{self.source_type}:{self.source_id}{loc}"


class SignalClaim(BaseModel):
    """A worker's declaration of the new exogenous evidence it will bring.

    This is what the admissibility gate adjudicates: does the promised signal
    actually come from *outside* the shared belief state?
    """

    claim: str = Field(..., description="what question this delegation resolves")
    expected_signal_source: str | None = Field(
        None, description="external source the worker will read, if any"
    )
    tool_names: list[str] = Field(default_factory=list)
    rationale: str = ""


class Delegation(BaseModel):
    """The four-tuple Φ = (Instruction, Context, Tools, Model) plus a SignalClaim.

    The orchestrator emits these; it may take no other environment action.
    """

    instruction: str
    context: list[str] = Field(default_factory=list, description="curated context for the worker")
    tools: list[str] = Field(default_factory=list, description="names of tools scoped to this worker")
    model: str | None = Field(None, description="model override; None -> default worker model")
    signal_claim: SignalClaim
    decision_key: str = Field(
        ..., description="the belief key this delegation is meant to resolve"
    )
    delegation_type: str = Field(
        ..., description="stable label for economics stats, e.g. 'verify_metric'"
    )


class Observation(BaseModel):
    """The structured result a worker returns to the orchestrator."""

    summary: str
    posterior: Posterior | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    tokens: int = 0
    latency_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None


class BeliefMessage(BaseModel):
    """A posterior-preserving message passed between hops (not prose)."""

    key: str = Field(..., description="belief key, e.g. 'headline_metric_matches'")
    claim: str
    posterior: Posterior
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    summary: str = Field("", description="human-readable gloss; NOT load-bearing")


class GateDecision(BaseModel):
    """The admissibility gate's verdict on a proposed delegation."""

    admit: bool
    cls: AdmissibilityClass
    reason: str
    # a-posteriori economics (filled in after the worker runs, if admitted)
    posterior_shift: float | None = None
    redundant: bool | None = None
