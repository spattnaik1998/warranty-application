"""The AuditReport — the product's deliverable.

A per-node verdict combining the four capability signals into one recommendation
with an honest confidence, plus run-level aggregates and the projected dollar
savings from collapsing what the audit found redundant. Rendering (CLI/HTML/JSON)
hangs off this pure-data model; HTML lives in ``warrant.report`` and is imported
lazily so ``audit()`` returns a usable object even without matplotlib.
"""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, Field

from warrant.schemas.belief import AdmissibilityClass


class Recommendation(str, Enum):
    KEEP = "KEEP"                # load-bearing or exogenous — earns its warrant
    COLLAPSE = "COLLAPSE"        # redundant reorganizer — fold into orchestrator
    REVIEW = "REVIEW"           # signals conflict — a human should look


class NodeFinding(BaseModel):
    """The audit's verdict on a single agent node."""

    node_id: str
    admissibility: AdmissibilityClass
    recommendation: Recommendation
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    # Evidence (each may be absent depending on annotation level).
    ablation_tested: bool = False
    ablation_value: float | None = None       # fraction of runs whose output changed
    mean_novelty: float | None = None          # 0 = pure restatement, 1 = all new
    mean_distortion_bits: float | None = None  # level-4 only
    tools: list[str] = Field(default_factory=list)

    # Economics.
    runs: int = 0
    mean_tokens: int = 0
    dollars_per_month: float = 0.0
    projected_savings_per_month: float = 0.0

    reason: str = ""

    @property
    def collapsible(self) -> bool:
        return self.recommendation is Recommendation.COLLAPSE


class AuditReport(BaseModel):
    """The full delegation audit over a set of runs."""

    graph_name: str = ""
    n_runs: int = 0
    findings: list[NodeFinding] = Field(default_factory=list)
    total_tokens: int = 0
    total_dollars_per_month: float = 0.0
    projected_savings_per_month: float = 0.0
    distortion_available: bool = False
    mean_chain_loss_bits: float = 0.0
    notes: list[str] = Field(default_factory=list)

    # -- serialization ------------------------------------------------------- #
    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def collapsible(self) -> list[NodeFinding]:
        return [f for f in self.findings if f.collapsible]

    # -- human-readable CLI table ------------------------------------------- #
    def to_cli(self) -> str:
        lines: list[str] = []
        lines.append(f"Warrant delegation audit — {self.graph_name}  ({self.n_runs} run(s))")
        lines.append("=" * 68)
        header = f"{'node':<16}{'verdict':<10}{'class':<13}{'abl':>6}{'nov':>6}{'$/mo':>10}"
        lines.append(header)
        lines.append("-" * 68)
        for f in self.findings:
            abl = "-" if f.ablation_value is None else f"{f.ablation_value:.2f}"
            nov = "-" if f.mean_novelty is None else f"{f.mean_novelty:.2f}"
            lines.append(
                f"{f.node_id:<16}{f.recommendation.value:<10}{f.admissibility.value:<13}"
                f"{abl:>6}{nov:>6}{f.dollars_per_month:>10.2f}"
            )
        lines.append("-" * 68)
        lines.append(
            f"Projected savings from collapsing {len(self.collapsible())} node(s): "
            f"${self.projected_savings_per_month:,.2f}/mo "
            f"of ${self.total_dollars_per_month:,.2f}/mo total."
        )
        if self.distortion_available:
            lines.append(f"Mean posterior chain-loss: {self.mean_chain_loss_bits:.3f} bits.")
        for note in self.notes:
            lines.append(f"note: {note}")
        for f in self.collapsible():
            lines.append(f"  → COLLAPSE {f.node_id}: {f.reason}")
        return "\n".join(lines)

    # -- HTML (lazy; requires warrant.report) -------------------------------- #
    def to_html(self, path: str | None = None) -> str:
        from warrant.report import render_html

        html = render_html(self)
        if path:
            from pathlib import Path

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(html, encoding="utf-8")
        return html

    def to_json_file(self, path: str) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")


__all__ = ["Recommendation", "NodeFinding", "AuditReport"]
