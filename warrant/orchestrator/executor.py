"""Worker executor: runs an admitted delegation and folds its signal into belief.

A worker is scoped to exactly the tools named in Φ. It executes them, then
elicits a posterior for the delegation's decision key from *its own* scoped
observation (not the global context) — modelling agent isolation, which is what
makes relay loss real. The observation's posterior shift is recorded in the
economics ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

from warrant.gate.novelty_audit import DelegationEconomics
from warrant.logging_setup import get_logger, log_event
from warrant.pipeline.steps import DecisionKey, PipelineContext
from warrant.schemas.belief import (
    AdmissibilityClass,
    Delegation,
    EvidenceRef,
    Posterior,
    SignalClaim,
)
from warrant.schemas.ledger import HopRecord

log = get_logger("orchestrator")


@dataclass
class PlanStep:
    """A unit of work the orchestrator may delegate (or collapse)."""

    phase: str  # 'discover' | 'fetch' | 'read' | 'verify' | 'compose'
    delegation_type: str
    tools: list[str]
    signal_source: str
    instruction: str
    decision_key: str | None = None
    dk: DecisionKey | None = None


def build_delegation(ctx: PipelineContext, step: PlanStep) -> Delegation:
    """Construct the four-tuple Φ plus a SignalClaim for a plan step."""
    return Delegation(
        instruction=step.instruction,
        context=[ctx.artifacts.get("title", "")] if step.phase != "discover" else [],
        tools=list(step.tools),
        model=None,
        signal_claim=SignalClaim(
            claim=step.instruction,
            expected_signal_source=step.signal_source if step.signal_source != "none" else None,
            tool_names=list(step.tools),
            rationale=f"phase={step.phase}",
        ),
        decision_key=step.decision_key or step.phase,
        delegation_type=step.delegation_type,
    )


def _run_tools(ctx: PipelineContext, step: PlanStep) -> tuple[str, list[EvidenceRef]]:
    """Execute the step's tools, updating ctx.artifacts. Returns (observation, refs)."""
    req = ctx.request
    obs_parts: list[str] = []
    refs: list[EvidenceRef] = []
    for name in step.tools:
        if step.phase == "discover" and name == "youtube_latest":
            result = ctx.run_tool(name, channel=req.youtube_channel)
            ids = result.artifacts.get("arxiv_ids", "").split(",")
            ctx.artifacts["arxiv_id"] = ids[0] if ids and ids[0] else ""
            ctx.artifacts["video_url"] = result.artifacts.get("url", "")
        elif step.phase == "discover" and name == "arxiv_search":
            result = ctx.run_tool(name, query=req.arxiv_query)
            ctx.artifacts["arxiv_id"] = result.artifacts["arxiv_id"]
        elif step.phase == "fetch":
            result = ctx.run_tool(name, arxiv_id=ctx.artifacts["arxiv_id"])
            ctx.artifacts["title"] = result.artifacts["title"]
            ctx.artifacts["authors"] = result.artifacts["authors"]
            ctx.artifacts["abstract"] = result.artifacts["abstract"]
        elif step.phase == "read":
            result = ctx.run_tool(name, arxiv_id=ctx.artifacts["arxiv_id"])
            ctx.artifacts["paper_text"] = result.artifacts["text"]
        elif step.phase == "verify":
            metric = step.decision_key.split("::", 1)[1] if step.decision_key else ""
            result = ctx.run_tool(name, arxiv_id=ctx.artifacts["arxiv_id"], metric=metric)
        else:
            result = ctx.run_tool(name)
        obs_parts.append(result.observation)
        refs.extend(result.evidence_refs)
    return "\n".join(obs_parts), refs


def run_delegation(
    ctx: PipelineContext,
    step: PlanStep,
    cls: AdmissibilityClass,
    economics: DelegationEconomics,
) -> HopRecord:
    """Execute an admitted delegation and return its telemetry."""
    observation, refs = _run_tools(ctx, step)

    # The 'paper_identified' belief can only be formed once the title is fetched.
    if step.phase == "fetch" and step.dk is None:
        title = ctx.artifacts.get("title", "")
        token = title.split()[0].lower() if title else "paper"
        step.dk = DecisionKey(
            key="paper_identified",
            claim=f"The briefing is about the paper titled '{title}'.",
            options=["correct", "wrong"],
            correct="correct",
            hints={"correct": [token, ctx.artifacts.get("arxiv_id", "")], "wrong": []},
        )
        step.decision_key = "paper_identified"

    prior: Posterior | None = None
    posterior: Posterior | None = None
    shift = 0.0
    redundant = False

    if step.dk is not None:
        prior = ctx.beliefs.posterior(step.dk.key)
        posterior = ctx.worker.classify_posterior(
            question=step.dk.claim,
            options=step.dk.options,
            context=observation,      # scoped to the worker's own observation
            hints=step.dk.hints,
        )
        shift = ctx.beliefs.update(step.dk.key, posterior, claim=step.dk.claim,
                                   evidence_refs=refs)
        redundant = economics.record(step.delegation_type, shift)

    hop = HopRecord(
        index=len(ctx.hops),
        delegation_type=step.delegation_type,
        cls=cls,
        admitted=True,
        decision_key=step.decision_key or step.phase,
        prior=prior,
        posterior=posterior,
        posterior_shift=shift,
        redundant=redundant,
    )
    ctx.hops.append(hop)
    log_event(log, "delegation executed", stage="execute", agent="worker",
              delegation_class=cls.value, tool=",".join(step.tools),
              posterior_shift=round(shift, 4), status="ok")
    return hop


def metric_keys_from_context(ctx: PipelineContext, max_metrics: int = 5) -> list[DecisionKey]:
    """Decision keys for each headline metric (available after fetch)."""
    from warrant.tools.fixtures import PAPERS

    arxiv_id = ctx.artifacts.get("arxiv_id", "")
    metrics = PAPERS.get(arxiv_id, {}).get("key_metrics", [])[:max_metrics]
    return [
        DecisionKey(
            key=f"metric::{m}",
            claim=f"The source reports the figure {m}.",
            options=["supported", "unsupported"],
            correct="supported",
            hints={"supported": [m], "unsupported": []},
        )
        for m in metrics
    ]
