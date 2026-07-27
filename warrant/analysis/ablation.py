"""Ablation-based delegation value (capability 2) — the headline feature.

The rigorous, framework-agnostic test of whether an agent earns its place:
re-run the graph with that node disabled and see whether the final answer
changes. A node whose removal never changes the output has **zero delegation
value** — it is pure posterior-preserving reorganization, safe to collapse, and
its tokens are pure waste. A node whose removal breaks the output is
load-bearing and must stay. This is what distinguishes a *redundant* reviewer
from a *load-bearing* writer even when both call no exogenous tool.

Requires a graph factory ``build_graph(disabled_nodes) -> compiled graph`` on
the instrumented app. Without it, ablation degrades gracefully to
``tested=False`` and the audit falls back to structural + novelty signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from warrant.logging_setup import get_logger, log_event
from warrant.trace.contract import content_digest

log = get_logger("warrant.analysis.ablation")


@dataclass
class AblationResult:
    """Outcome of disabling one node and replaying every recorded input."""

    node_id: str
    tested: bool
    n_runs: int = 0
    changed: int = 0

    @property
    def delegation_value(self) -> float:
        """Fraction of runs whose final output changed when the node was disabled.

        0.0 ⇒ the node changed nothing (collapsible). 1.0 ⇒ always load-bearing.
        """
        return self.changed / self.n_runs if self.n_runs else 0.0

    @property
    def collapsible(self) -> bool:
        return self.tested and self.n_runs > 0 and self.changed == 0


def ablate_node(app: Any, node_id: str, baseline_by_run: dict[str, str]) -> AblationResult:
    """Disable ``node_id``, replay recorded inputs, and diff against baselines."""
    if app is None or getattr(app, "build_graph", None) is None:
        return AblationResult(node_id=node_id, tested=False)

    try:
        disabled_graph = app.build_graph(frozenset({node_id}))
    except Exception as exc:  # never swallow — record and degrade
        log_event(log, "build_graph failed", stage="ablation", node_id=node_id, status="error", error=str(exc))
        return AblationResult(node_id=node_id, tested=False)

    n = 0
    changed = 0
    for run_id, input, config in app.replays:
        baseline = baseline_by_run.get(run_id)
        if baseline is None:
            continue
        n += 1
        try:
            ablated_output = app.replay_output_blocking(disabled_graph, input, config)
        except Exception:
            # Disabling the node broke the run — it is load-bearing by definition.
            changed += 1
            continue
        if content_digest(ablated_output) != content_digest(baseline):
            changed += 1
    log_event(
        log,
        "ablation complete",
        stage="ablation",
        node_id=node_id,
        status="ok",
        n_runs=n,
        changed=changed,
    )
    return AblationResult(node_id=node_id, tested=True, n_runs=n, changed=changed)


def ablation_audit(app: Any, traces: list[Any], node_ids: list[str]) -> dict[str, AblationResult]:
    """Ablate each of ``node_ids`` against the baseline outputs from ``traces``."""
    baseline_by_run = {t.run_id: t.final_output for t in traces}
    return {node_id: ablate_node(app, node_id, baseline_by_run) for node_id in node_ids}


__all__ = ["AblationResult", "ablate_node", "ablation_audit"]
