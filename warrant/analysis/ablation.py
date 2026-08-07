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

**The nondeterminism floor.** A diff between two runs of an LLM graph only means
something if the graph reproduces itself when *nothing* is removed. Before
ablating anything, this module replays every recorded input through
``build_graph(frozenset())`` and checks the output still matches. If it doesn't,
the graph is nondeterministic, every ablation diff on it is noise, and the audit
says so and caps its confidence rather than reporting a verdict it cannot support.
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
    changed: int = 0        # replays whose final output differed
    errored: int = 0        # replays that raised — load-bearing by definition
    baseline_stable: bool | None = None  # None = could not be checked

    @property
    def affected(self) -> int:
        """Replays where removing the node demonstrably mattered."""
        return self.changed + self.errored

    @property
    def delegation_value(self) -> float:
        """Fraction of runs where disabling the node changed or broke the output.

        0.0 ⇒ the node changed nothing (collapsible). 1.0 ⇒ always load-bearing.
        """
        return self.affected / self.n_runs if self.n_runs else 0.0

    @property
    def collapsible(self) -> bool:
        return self.tested and self.n_runs > 0 and self.affected == 0


def _can_ablate(app: Any) -> bool:
    return app is not None and getattr(app, "build_graph", None) is not None


def baseline_stability(app: Any, baseline_by_run: dict[str, str]) -> bool | None:
    """Replay every input with nothing disabled; True iff outputs reproduce.

    Returns ``None`` when the check could not be run at all (no factory, or the
    factory itself failed) — an absent answer, never a reassuring one.
    """
    if not _can_ablate(app):
        return None
    try:
        graph = app.build_graph(frozenset())
    except Exception as exc:  # never swallow — record and degrade
        log_event(
            log, "build_graph failed on baseline", stage="ablation",
            status="error", error=str(exc),
        )
        return None

    checked = 0
    for run_id, input, config in app.replays:
        baseline = baseline_by_run.get(run_id)
        if baseline is None:
            continue
        checked += 1
        try:
            replayed = app.replay_output_blocking(graph, input, config)
        except Exception as exc:
            log_event(
                log, "baseline replay failed", stage="ablation",
                run_id=run_id, status="error", error=str(exc),
            )
            return False
        if content_digest(replayed) != content_digest(baseline):
            log_event(
                log, "graph is nondeterministic", stage="ablation",
                run_id=run_id, status="unstable",
            )
            return False
    return True if checked else None


def ablate_node(
    app: Any,
    node_id: str,
    baseline_by_run: dict[str, str],
    baseline_stable: bool | None = None,
) -> AblationResult:
    """Disable ``node_id``, replay recorded inputs, and diff against baselines."""
    if not _can_ablate(app):
        return AblationResult(node_id=node_id, tested=False)

    try:
        disabled_graph = app.build_graph(frozenset({node_id}))
    except Exception as exc:  # never swallow — record and degrade
        log_event(log, "build_graph failed", stage="ablation", node_id=node_id, status="error", error=str(exc))
        return AblationResult(node_id=node_id, tested=False)

    n = 0
    changed = 0
    errored = 0
    for run_id, input, config in app.replays:
        baseline = baseline_by_run.get(run_id)
        if baseline is None:
            continue
        n += 1
        try:
            ablated_output = app.replay_output_blocking(disabled_graph, input, config)
        except Exception:
            # Disabling the node broke the run — load-bearing, but for a different
            # reason than "the answer changed". Counted separately so a rate limit
            # is never silently reported as delegation value.
            errored += 1
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
        errored=errored,
    )
    return AblationResult(
        node_id=node_id,
        tested=True,
        n_runs=n,
        changed=changed,
        errored=errored,
        baseline_stable=baseline_stable,
    )


def ablation_audit(app: Any, traces: list[Any], node_ids: list[str]) -> dict[str, AblationResult]:
    """Ablate each of ``node_ids`` against the baseline outputs from ``traces``.

    The un-ablated baseline replay runs once for the whole audit, not once per
    node, so the nondeterminism check costs one extra pass over the inputs.
    """
    baseline_by_run = {t.run_id: t.final_output for t in traces}
    stable = baseline_stability(app, baseline_by_run) if node_ids else None
    return {
        node_id: ablate_node(app, node_id, baseline_by_run, stable) for node_id in node_ids
    }


__all__ = ["AblationResult", "ablate_node", "ablation_audit", "baseline_stability"]
