"""The AuditReport — the product's deliverable.

A per-node verdict combining the four capability signals into one recommendation
with a confidence that tracks the evidence behind it, plus run-level aggregates
and the dollar savings from collapsing what the audit found redundant. Rendering
(CLI/HTML/JSON) hangs off this pure-data model; HTML lives in ``warrant.report``
and is imported lazily so ``audit()`` returns a usable object either way.

Two rules govern the economics fields. The measured unit is **dollars per 1,000
runs** — it follows directly from observed token usage and assumes nothing.
``dollars_per_month`` is populated only when the caller *declared* a traffic
volume, and every surface that prints it says "declared", because Warrant does
not know how often your graph runs. When cost cannot be measured at all (a static
scan), ``economics_available`` is False and the money is suppressed rather than
rendered as ``$0.00``.
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
    ablation_runs: int = 0                     # n behind ablation_value
    ablation_errors: int = 0                   # replays that crashed rather than differed
    baseline_stable: bool | None = None        # did an un-ablated replay reproduce?
    mean_novelty: float | None = None          # 0 = pure restatement, 1 = all new
    mean_distortion_bits: float | None = None  # level-4 only
    tools: list[str] = Field(default_factory=list)

    # Economics. Monthly figures are None unless a volume was declared.
    runs: int = 0
    mean_tokens: int = 0
    model: str | None = None                   # what this node was priced at
    dollars_per_1k_runs: float = 0.0
    dollars_per_month: float | None = None
    projected_savings_per_1k_runs: float = 0.0
    projected_savings_per_month: float | None = None

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

    # Economics. ``economics_available`` is False when cost cannot be measured
    # (e.g. a static scan), and every renderer then omits the money entirely.
    economics_available: bool = True
    runs_per_month: int | None = None          # declared by the caller, never guessed
    total_dollars_per_1k_runs: float = 0.0
    total_dollars_per_month: float | None = None
    projected_savings_per_1k_runs: float = 0.0
    projected_savings_per_month: float | None = None

    distortion_available: bool = False
    mean_chain_loss_bits: float | None = None  # None = not measured, not "zero"
    notes: list[str] = Field(default_factory=list)

    # -- serialization ------------------------------------------------------- #
    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def collapsible(self) -> list[NodeFinding]:
        return [f for f in self.findings if f.collapsible]

    # -- economics phrasing (shared by CLI and HTML) ------------------------- #
    def savings_sentence(self) -> str:
        """One sentence of savings, in whichever unit the evidence supports."""
        n = len(self.collapsible())
        if not self.economics_available:
            return f"{n} reorganizer candidate(s) to investigate — run the graph to price them."
        if self.runs_per_month is not None:
            return (
                f"Collapsing {n} redundant agent(s) saves "
                f"${self.projected_savings_per_month or 0.0:,.2f}/mo at "
                f"{self.runs_per_month:,} runs/month (declared), of "
                f"${self.total_dollars_per_month or 0.0:,.2f}/mo total."
            )
        share = (
            self.projected_savings_per_1k_runs / self.total_dollars_per_1k_runs
            if self.total_dollars_per_1k_runs
            else 0.0
        )
        return (
            f"Collapsing {n} redundant agent(s) saves "
            f"${self.projected_savings_per_1k_runs:,.2f} per 1,000 runs "
            f"({share:.0%} of measured spend)."
        )

    @property
    def money_column(self) -> str:
        return "$/mo" if self.runs_per_month is not None else "$/1k runs"

    def _money(self, finding: NodeFinding) -> float:
        if self.runs_per_month is not None:
            return finding.dollars_per_month or 0.0
        return finding.dollars_per_1k_runs

    # -- human-readable CLI table ------------------------------------------- #
    def to_cli(self) -> str:
        width = 74
        lines: list[str] = []
        lines.append(f"Warrant delegation audit — {self.graph_name}  ({self.n_runs} run(s))")
        lines.append("=" * width)
        money = self.money_column
        header = (
            f"{'node':<16}{'verdict':<10}{'class':<13}"
            f"{'ablation':>14}{'nov':>6}{money:>15}"
        )
        lines.append(header)
        lines.append("-" * width)
        for f in self.findings:
            if f.ablation_value is None:
                abl = "-"
            else:
                abl = f"{f.ablation_value:.2f} (n={f.ablation_runs})"
            nov = "-" if f.mean_novelty is None else f"{f.mean_novelty:.2f}"
            cash = "-" if not self.economics_available else f"{self._money(f):,.2f}"
            lines.append(
                f"{f.node_id:<16}{f.recommendation.value:<10}{f.admissibility.value:<13}"
                f"{abl:>14}{nov:>6}{cash:>15}"
            )
        lines.append("-" * width)
        lines.append(self.savings_sentence())
        if self.distortion_available and self.mean_chain_loss_bits is not None:
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
