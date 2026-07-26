"""The audit engine: fuse the four capability signals into an AuditReport.

``run_audit`` is what ``warrant.audit()`` calls. It runs every analyzer that the
available evidence supports and combines them per node into a single
recommendation with an honest confidence — richer annotation (a graph factory
for ablation, decision points for distortion) yields higher-confidence verdicts,
and their absence degrades gracefully rather than fabricating certainty.
"""

from __future__ import annotations

from typing import Any

from warrant.analysis.ablation import AblationResult, ablation_audit
from warrant.analysis.cost import estimate_cost
from warrant.analysis.distortion import distortion_audit
from warrant.analysis.novelty import novelty_audit
from warrant.analysis.report import AuditReport, NodeFinding, Recommendation
from warrant.analysis.structural import structural_audit
from warrant.config import get_settings
from warrant.schemas.belief import AdmissibilityClass
from warrant.trace.contract import RunTrace

_NOVELTY_EPSILON = 0.15
_RUNS_PER_MONTH = 30_000


def _mean_tokens(traces: list[RunTrace], node_id: str) -> tuple[int, int]:
    """Return (mean tokens per run, number of runs) for a node."""
    tokens = [n.outcome.tokens for t in traces for n in t.nodes if n.node_id == node_id]
    if not tokens:
        return 0, 0
    return round(sum(tokens) / len(tokens)), len(tokens)


def _decide(
    admissibility: AdmissibilityClass,
    ablation: AblationResult,
    mean_novelty: float,
) -> tuple[Recommendation, float, str]:
    """Fuse signals into (recommendation, confidence, reason)."""
    # Exogenous nodes earn their warrant by construction.
    if admissibility is not AdmissibilityClass.REORGANIZER:
        return (
            Recommendation.KEEP,
            0.9,
            f"{admissibility.value}: calls an exogenous tool that injects new signal.",
        )

    # Reorganizer: prefer the rigorous ablation verdict when available.
    if ablation.tested and ablation.n_runs > 0:
        if ablation.collapsible:
            return (
                Recommendation.COLLAPSE,
                0.92,
                "reorganizer with zero delegation value: disabling it changed the "
                f"final answer in 0/{ablation.n_runs} runs. Pure posterior-preserving "
                "restatement — collapse into the orchestrator.",
            )
        return (
            Recommendation.KEEP,
            0.8,
            "load-bearing transform: calls no exogenous tool, but disabling it changed "
            f"the answer in {ablation.changed}/{ablation.n_runs} runs, so it does "
            "necessary work (e.g. final synthesis). Keep, but it cannot be parallelized away.",
        )

    # No ablation: fall back to the novelty proxy, at lower confidence.
    if mean_novelty < _NOVELTY_EPSILON:
        return (
            Recommendation.COLLAPSE,
            0.5,
            f"reorganizer with low output novelty ({mean_novelty:.2f} < {_NOVELTY_EPSILON}): "
            "its output largely restates prior context. Provide a build_graph factory to "
            "confirm by ablation.",
        )
    return (
        Recommendation.REVIEW,
        0.4,
        f"reorganizer that adds novel text ({mean_novelty:.2f}) but calls no exogenous "
        "tool. Cannot prove redundancy without ablation — provide a build_graph factory.",
    )


def run_audit(traces: list[RunTrace], app: Any = None) -> AuditReport:
    """Analyze recorded runs and return the delegation :class:`AuditReport`."""
    settings = get_settings()
    if not traces:
        return AuditReport(notes=["No runs recorded — invoke the instrumented app first."])

    structure = structural_audit(traces)
    novelty = novelty_audit(traces, epsilon=_NOVELTY_EPSILON)
    distortion = distortion_audit(traces)

    # Ablation is the headline signal but needs a graph factory.
    reorganizers = [nid for nid, s in structure.items() if s.is_reorganizer]
    ablation = ablation_audit(app, traces, reorganizers) if app is not None else {}

    notes: list[str] = []
    token_modes = {t.labels.get("tokens", "estimated") for t in traces}
    if token_modes == {"measured"}:
        notes.append("Cost figures use measured token usage (LangChain usage_metadata).")
    else:
        notes.append(
            "Cost figures use estimated token counts (~4 chars/token) for at least "
            "some nodes — verdicts are unaffected, but treat $/mo as directional. "
            "Emit usage_metadata from your nodes for billing-accurate cost."
        )
    if app is None or getattr(app, "build_graph", None) is None:
        notes.append(
            "Ablation unavailable: pass build_graph(disabled_nodes) to instrument() to "
            "unlock rigorous delegation-value proof (verdicts on reorganizers are lower confidence)."
        )
    if not distortion.available:
        notes.append(
            "Posterior-distortion analysis inactive: annotate belief checkpoints with "
            "warrant.decision(...) to unlock the theorem-grade proof and the ledger."
        )

    model = settings.worker_model
    findings: list[NodeFinding] = []
    total_dollars = 0.0
    saved = 0.0
    for node_id, struct in structure.items():
        mean_tok, runs = _mean_tokens(traces, node_id)
        cost = estimate_cost(mean_tok, runs, model=model, runs_per_month=_RUNS_PER_MONTH)
        total_dollars += cost.dollars_per_month

        abl = ablation.get(node_id, AblationResult(node_id=node_id, tested=False))
        mean_nov = novelty[node_id].mean_novelty if node_id in novelty else 0.0
        dist = distortion.per_node.get(node_id)

        rec, conf, reason = _decide(struct.admissibility, abl, mean_nov)
        savings = cost.dollars_per_month if rec is Recommendation.COLLAPSE else 0.0
        saved += savings

        findings.append(
            NodeFinding(
                node_id=node_id,
                admissibility=struct.admissibility,
                recommendation=rec,
                confidence=conf,
                ablation_tested=abl.tested,
                ablation_value=abl.delegation_value if abl.tested else None,
                mean_novelty=round(mean_nov, 3) if node_id in novelty else None,
                mean_distortion_bits=round(dist.mean_kl, 4) if dist else None,
                tools=sorted(struct.tool_names),
                runs=runs,
                mean_tokens=mean_tok,
                dollars_per_month=round(cost.dollars_per_month, 2),
                projected_savings_per_month=round(savings, 2),
                reason=reason,
            )
        )

    return AuditReport(
        graph_name=traces[0].graph_name,
        n_runs=len(traces),
        findings=findings,
        total_tokens=sum(t.total_tokens for t in traces),
        total_dollars_per_month=round(total_dollars, 2),
        projected_savings_per_month=round(saved, 2),
        distortion_available=distortion.available,
        mean_chain_loss_bits=round(distortion.mean_chain_loss_bits, 4),
        notes=notes,
    )


__all__ = ["run_audit", "AuditReport", "NodeFinding", "Recommendation"]
