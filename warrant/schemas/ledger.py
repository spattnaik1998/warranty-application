"""Ledger schemas: per-hop and per-run records for the Delegation Ledger probe."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from warrant.schemas.belief import AdmissibilityClass, Posterior


class Condition(str, Enum):
    """The matched experimental conditions (mirroring the paper's design)."""

    CENTRALIZED = "A_centralized"      # single call with all evidence pre-gathered
    GOVERNED = "B_governed"            # admissibility-gated network
    NAIVE = "B_naive"                  # gate disabled: free reorganizer relay
    SIGNAL_STARVED = "C_signal_starved"  # a genuine retrieval tool removed


class HopRecord(BaseModel):
    """One delegation hop's telemetry."""

    index: int
    delegation_type: str
    cls: AdmissibilityClass
    admitted: bool
    decision_key: str
    prior: Posterior | None = None
    posterior: Posterior | None = None
    # Posterior shift this hop induced on the belief state (bits, KL).
    posterior_shift: float = 0.0
    # Distortion vs the centralized baseline posterior for this key.
    distortion_vs_centralized: float = 0.0
    redundant: bool = False
    tokens: int = 0
    latency_ms: int = 0
    error: str | None = None


class RunRecord(BaseModel):
    """Telemetry for a full task run under one condition."""

    task_id: str
    condition: Condition
    hops: list[HopRecord] = Field(default_factory=list)
    # Verifiable-fact accuracy in [0, 1] scored by the deterministic oracle.
    accuracy: float = 0.0
    # Aggregate communication loss (sum of per-key distortion vs centralized).
    communication_loss: float = 0.0
    delegation_depth: int = 0
    admitted_delegations: int = 0
    rejected_delegations: int = 0
    redundant_delegations: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
    flagged_for_review: int = 0

    def summary_row(self) -> dict[str, float | str | int]:
        return {
            "condition": self.condition.value,
            "accuracy": round(self.accuracy, 3),
            "comm_loss": round(self.communication_loss, 4),
            "depth": self.delegation_depth,
            "admitted": self.admitted_delegations,
            "rejected": self.rejected_delegations,
            "redundant": self.redundant_delegations,
            "tokens": self.total_tokens,
        }
