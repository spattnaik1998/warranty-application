"""The audit engine: fuse the four capability signals into an AuditReport.

``run_audit`` is what ``warrant.audit()`` calls. It runs every analyzer that the
available evidence supports and combines them per node into a single
recommendation whose confidence is a function of that evidence — more runs, a
stable baseline, a declared output key and declared tools all raise it, and their
absence lowers it rather than being papered over.

Three rules keep the numbers defensible:

* **Confidence follows the evidence.** A COLLAPSE verdict from zero changes in
  *n* runs is scored by the rule of three (the 95% upper bound on the true
  change-rate is ≈ 3/n), so one run never renders as near-certainty.
* **Every node is priced at the model it actually called**, read from the trace.
  Only nodes whose model the framework never reported fall back to the configured
  default, and the report names them.
* **Volume is an input, never an assumption.** The measured unit is dollars per
  1,000 runs. A monthly figure appears only when the caller declares their
  traffic, and is always labelled as declared.
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

# Confidence bounds. The floor keeps a single clean run from reading as "no
# signal at all"; the ceiling reflects that ablation is a sample, never a proof.
_CONF_FLOOR = 0.2
_CONF_CEILING = 0.95

# Caps applied when a specific piece of evidence is missing or compromised.
_CAP_NO_TOOLS = 0.5        # nothing declared or observed — everything looks redundant
_CAP_NO_OUTPUT_KEY = 0.6   # diffing the whole state biases every verdict to KEEP
_CAP_UNSTABLE = 0.4        # the graph does not reproduce itself; diffs are noise


def _novelty_epsilon() -> float:
    """Redundancy threshold for the level-3 proxy (WARRANT_NOVELTY_EPSILON)."""
    return get_settings().novelty_epsilon


def _collapse_confidence(n_runs: int) -> float:
    """Confidence in COLLAPSE after ``n_runs`` replays that changed nothing.

    Uses the rule of three: having seen zero events in n trials, the 95% upper
    bound on the true rate is ≈ 3/n. One run is weak evidence and scores like it.
    """
    if n_runs <= 0:
        return 0.0
    return max(_CONF_FLOOR, min(_CONF_CEILING, 1.0 - 3.0 / n_runs))


def _keep_confidence(delegation_value: float) -> float:
    """Confidence in KEEP after observing the node change the answer.

    Asymmetric with COLLAPSE on purpose: a single observed change *demonstrates*
    the node does work, where zero changes only bounds how often it might.
    """
    return min(_CONF_CEILING, 0.7 + 0.25 * delegation_value)


def _mean(values: list[int]) -> int:
    return round(sum(values) / len(values)) if values else 0


def _decide(
    admissibility: AdmissibilityClass,
    ablation: AblationResult,
    mean_novelty: float,
    cap: float,
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
                min(cap, _collapse_confidence(ablation.n_runs)),
                "reorganizer with zero delegation value: disabling it changed the "
                f"final answer in 0/{ablation.n_runs} runs. Pure posterior-preserving "
                "restatement — collapse into the orchestrator.",
            )
        crashed = (
            f" ({ablation.errored} of those raised rather than differed)"
            if ablation.errored
            else ""
        )
        return (
            Recommendation.KEEP,
            min(cap, _keep_confidence(ablation.delegation_value)),
            "load-bearing transform: calls no exogenous tool, but disabling it changed "
            f"the answer in {ablation.affected}/{ablation.n_runs} runs{crashed}, so it "
            "does necessary work (e.g. final synthesis). Keep, but it cannot be "
            "parallelized away.",
        )

    # No ablation: fall back to the novelty proxy, at lower confidence.
    epsilon = _novelty_epsilon()
    if mean_novelty < epsilon:
        return (
            Recommendation.COLLAPSE,
            min(cap, 0.5),
            f"reorganizer with low output novelty ({mean_novelty:.2f} < {epsilon}): "
            "its output largely restates prior context. Provide a build_graph factory to "
            "confirm by ablation.",
        )
    return (
        Recommendation.REVIEW,
        min(cap, 0.4),
        f"reorganizer that adds novel text ({mean_novelty:.2f}) but calls no exogenous "
        "tool. Cannot prove redundancy without ablation — provide a build_graph factory.",
    )


def _node_model(outcomes: list[Any], default: str) -> tuple[str | None, bool, bool]:
    """Pick the model a node was billed at: ``(model, measured, mixed)``.

    ``measured`` is False when the framework never reported one and we had to fall
    back to the configured default — a fact the report discloses.
    """
    by_model: dict[str, int] = {}
    mixed = False
    for out in outcomes:
        if out.model:
            by_model[out.model] = by_model.get(out.model, 0) + max(out.tokens, 1)
        mixed = mixed or out.mixed_models
    if by_model:
        return max(by_model, key=lambda name: by_model[name]), True, mixed or len(by_model) > 1
    billable = any(o.token_source != "none" and o.tokens for o in outcomes)
    return (default if billable else None), not billable, mixed


def run_audit(
    traces: list[RunTrace],
    app: Any = None,
    runs_per_month: int | None = None,
) -> AuditReport:
    """Analyze recorded runs and return the delegation :class:`AuditReport`."""
    settings = get_settings()
    if not traces:
        return AuditReport(notes=["No runs recorded — invoke the instrumented app first."])

    if runs_per_month is None:
        runs_per_month = settings.runs_per_month

    structure = structural_audit(traces)
    novelty = novelty_audit(traces, epsilon=_novelty_epsilon())
    distortion = distortion_audit(traces)

    # -- what evidence do we actually have? ---------------------------------- #
    tools_observed = any(n.tool_calls for t in traces for n in t.nodes)
    output_key = getattr(app, "output_key", None)
    can_ablate = app is not None and getattr(app, "build_graph", None) is not None

    # Ablation is the headline signal but needs a graph factory.
    reorganizers = [nid for nid, s in structure.items() if s.is_reorganizer]
    ablation = ablation_audit(app, traces, reorganizers) if app is not None else {}
    baseline_stable = next(
        (a.baseline_stable for a in ablation.values() if a.baseline_stable is not None), None
    )

    cap = 1.0
    notes: list[str] = []
    # Ordered so the verdict-inverting problems are read first.
    if not tools_observed:
        cap = min(cap, _CAP_NO_TOOLS)
        notes.append(
            "No tool calls were declared or observed, so every node classifies as a "
            "REORGANIZER by default and these verdicts are capped at low confidence. "
            "Pass node_tools={node: [tool, ...]} to instrument() — or emit tool calls "
            "your framework reports — to get a real structural audit."
        )
    if can_ablate and not output_key:
        cap = min(cap, _CAP_NO_OUTPUT_KEY)
        notes.append(
            "No output_key was declared, so ablation diffs the whole final state "
            "including intermediate fields. That biases every verdict toward KEEP; "
            "pass output_key='<your answer field>' to instrument() for a true diff."
        )
    if baseline_stable is False:
        cap = min(cap, _CAP_UNSTABLE)
        notes.append(
            "The graph did not reproduce its own recorded output when replayed with "
            "nothing disabled, so it is nondeterministic and ablation diffs cannot be "
            "distinguished from run-to-run noise. Verdicts are capped accordingly — fix "
            "determinism (temperature 0, cached retrieval) before trusting them."
        )

    # -- economics ----------------------------------------------------------- #
    findings: list[NodeFinding] = []
    total_per_1k = 0.0
    saved_per_1k = 0.0
    total_monthly = 0.0
    saved_monthly = 0.0
    estimated_nodes: list[str] = []
    default_priced: list[str] = []
    mixed_nodes: list[str] = []
    fallback_priced: list[str] = []
    any_measured = False

    for node_id, struct in structure.items():
        outcomes = [n.outcome for t in traces for n in t.nodes if n.node_id == node_id]
        runs = len(outcomes)
        mean_tok = _mean([o.tokens for o in outcomes])
        mean_prompt = _mean([o.prompt_tokens for o in outcomes])
        mean_completion = _mean([o.completion_tokens for o in outcomes])

        if any(o.token_source == "estimated" for o in outcomes):
            estimated_nodes.append(node_id)
        any_measured = any_measured or any(o.token_source == "measured" for o in outcomes)

        model, model_measured, mixed = _node_model(outcomes, settings.worker_model)
        if not model_measured and model:
            default_priced.append(node_id)
        if mixed:
            mixed_nodes.append(node_id)

        cost = estimate_cost(
            mean_tok,
            model,
            prompt_tokens=mean_prompt,
            completion_tokens=mean_completion,
            runs_per_month=runs_per_month,
        )
        if cost.fallback_price:
            fallback_priced.append(node_id)
        total_per_1k += cost.dollars_per_1k_runs
        total_monthly += cost.dollars_per_month or 0.0

        abl = ablation.get(node_id, AblationResult(node_id=node_id, tested=False))
        mean_nov = novelty[node_id].mean_novelty if node_id in novelty else 0.0
        dist = distortion.per_node.get(node_id)

        rec, conf, reason = _decide(struct.admissibility, abl, mean_nov, cap)
        collapsing = rec is Recommendation.COLLAPSE
        saving_1k = cost.dollars_per_1k_runs if collapsing else 0.0
        saving_month = (cost.dollars_per_month or 0.0) if collapsing else 0.0
        saved_per_1k += saving_1k
        saved_monthly += saving_month

        findings.append(
            NodeFinding(
                node_id=node_id,
                admissibility=struct.admissibility,
                recommendation=rec,
                confidence=round(conf, 2),
                ablation_tested=abl.tested,
                ablation_value=abl.delegation_value if abl.tested else None,
                ablation_runs=abl.n_runs,
                ablation_errors=abl.errored,
                baseline_stable=abl.baseline_stable,
                mean_novelty=round(mean_nov, 3) if node_id in novelty else None,
                mean_distortion_bits=round(dist.mean_kl, 4) if dist else None,
                tools=sorted(struct.tool_names),
                runs=runs,
                mean_tokens=mean_tok,
                model=model,
                dollars_per_1k_runs=round(cost.dollars_per_1k_runs, 4),
                dollars_per_month=(
                    None if cost.dollars_per_month is None else round(cost.dollars_per_month, 2)
                ),
                projected_savings_per_1k_runs=round(saving_1k, 4),
                projected_savings_per_month=(
                    None if runs_per_month is None else round(saving_month, 2)
                ),
                reason=reason,
            )
        )

    # -- provenance notes ---------------------------------------------------- #
    if estimated_nodes:
        notes.append(
            "Cost is estimated (~4 chars/token) for node(s) that ran a model without "
            f"reporting token usage: {', '.join(sorted(estimated_nodes))}. Verdicts are "
            "unaffected; emit usage metadata from those nodes for billing-accurate cost "
            "(other nodes are exact)."
        )
    elif any_measured:
        notes.append("Cost figures use measured token usage from your model responses.")
    if default_priced:
        notes.append(
            "No model name was reported for node(s) "
            f"{', '.join(sorted(default_priced))}, so they are priced at the configured "
            f"default ({settings.worker_model}). If they call a different model, their "
            "dollar figures are wrong — set WARRANT_WORKER_MODEL or emit response metadata."
        )
    if fallback_priced:
        notes.append(
            "No published rate is known for the model on node(s) "
            f"{', '.join(sorted(fallback_priced))}; a conservative default rate was used. "
            "Set WARRANT_PRICE_<MODEL>_IN / _OUT to price them exactly."
        )
    if mixed_nodes:
        notes.append(
            f"Node(s) {', '.join(sorted(mixed_nodes))} called more than one model in a "
            "single execution; each is priced at whichever model spent the most tokens."
        )
    if runs_per_month is None:
        notes.append(
            "Dollar figures are per 1,000 runs — the unit Warrant can measure. Declare "
            "your production volume (--runs-per-month N, or warrant.audit(runs_per_month=N)) "
            "to project a monthly figure."
        )
    if not can_ablate:
        notes.append(
            "Ablation unavailable: pass build_graph(disabled_nodes) to instrument() to "
            "unlock rigorous delegation-value proof (verdicts on reorganizers are lower confidence)."
        )
    if not distortion.available:
        notes.append(
            "Posterior-distortion analysis inactive: annotate belief checkpoints with "
            "warrant.decision(...) to unlock the theorem-grade proof and the ledger."
        )

    return AuditReport(
        graph_name=traces[0].graph_name,
        n_runs=len(traces),
        findings=findings,
        total_tokens=sum(t.total_tokens for t in traces),
        economics_available=True,
        runs_per_month=runs_per_month,
        total_dollars_per_1k_runs=round(total_per_1k, 4),
        total_dollars_per_month=None if runs_per_month is None else round(total_monthly, 2),
        projected_savings_per_1k_runs=round(saved_per_1k, 4),
        projected_savings_per_month=None if runs_per_month is None else round(saved_monthly, 2),
        distortion_available=distortion.available,
        mean_chain_loss_bits=(
            round(distortion.mean_chain_loss_bits, 4) if distortion.available else None
        ),
        notes=notes,
    )


__all__ = ["run_audit", "AuditReport", "NodeFinding", "Recommendation"]
