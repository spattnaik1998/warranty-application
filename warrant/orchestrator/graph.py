"""The governed two-move orchestrator, wired as a LangGraph state machine.

The orchestrator may only **Delegate** (route to the executor) or **Finish**
(route to END). It never calls a tool directly. Every proposed delegation is
adjudicated by the admissibility gate: exogenous injectors and non-redundant
validators are admitted; reorganizer work (composition) is *collapsed* into the
orchestrator's own single call rather than delegated — the concrete embodiment
of "a hop that adds no new signal is dominated by one centralized decision".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from warrant.belief.distortion import communication_loss
from warrant.gate import DelegationEconomics
from warrant.gate.admissibility import gate_decision
from warrant.logging_setup import get_logger, log_event
from warrant.orchestrator.escalation import flag_for_review
from warrant.orchestrator.executor import (
    PlanStep,
    build_delegation,
    metric_keys_from_context,
    run_delegation,
)
from warrant.pipeline.brief_pipeline import _evidence_context, ground_and_graycheck
from warrant.pipeline.steps import DecisionKey, PipelineContext, compose_sections, new_context
from warrant.schemas.belief import AdmissibilityClass, EvidenceRef
from warrant.schemas.ledger import HopRecord
from warrant.schemas.tasks import BriefRequest, BriefResult
from warrant.tools import REGISTRY

log = get_logger("orchestrator")

_MAX_STEPS = 64


class _State(TypedDict, total=False):
    ctx: PipelineContext
    econ: DelegationEconomics
    plan: list[PlanStep]
    dks: list[DecisionKey]
    current_step: PlanStep | None
    current_cls: AdmissibilityClass | None
    verify_expanded: bool
    gate_log: list[dict[str, Any]]
    result: BriefResult | None
    steps_taken: int


@dataclass
class GovernedResult:
    """Everything the ledger and the CLI need from a governed run."""

    brief: BriefResult
    ctx: PipelineContext
    econ: DelegationEconomics
    gate_log: list[dict[str, Any]]
    hops: list[HopRecord] = field(default_factory=list)

    @property
    def admitted(self) -> int:
        return sum(1 for g in self.gate_log if g["admit"])

    @property
    def rejected(self) -> int:
        return sum(1 for g in self.gate_log if not g["admit"])

    @property
    def redundant(self) -> int:
        return sum(1 for g in self.gate_log if g.get("redundant"))


def _initial_plan(ctx: PipelineContext) -> list[PlanStep]:
    plan: list[PlanStep] = []
    req = ctx.request
    if not ctx.artifacts.get("arxiv_id"):
        if req.youtube_channel:
            plan.append(PlanStep("discover", "discover_youtube", ["youtube_latest"],
                                 "youtube", "Find the newest video and its papers."))
        elif req.arxiv_query:
            plan.append(PlanStep("discover", "discover_search", ["arxiv_search"],
                                 "arxiv-db", "Search arXiv for the target paper."))
    plan.append(PlanStep("fetch", "fetch_metadata", ["arxiv_fetch"],
                         "arxiv-db", "Fetch the paper's metadata."))
    if ctx.tool_available("pdf_read"):
        plan.append(PlanStep("read", "read_paper", ["pdf_read"],
                             "paper-text", "Read the paper's full text."))
    return plan


def _finalize(state: _State) -> BriefResult:
    """Orchestrator's own reorganization: compose + ground-check (collapsed)."""
    ctx = state["ctx"]
    dks = state.get("dks", [])

    # Resolve any decision key that never received an exogenous signal, inline,
    # from whatever global evidence exists (empty under signal-starvation).
    evidence = _evidence_context(ctx)
    for dk in dks:
        if ctx.beliefs.get(dk.key) is None:
            posterior = ctx.worker.classify_posterior(
                question=dk.claim, options=dk.options, context=evidence, hints=dk.hints)
            shift = ctx.beliefs.update(dk.key, posterior, claim=dk.claim,
                                       evidence_refs=[EvidenceRef(source_type="pdf",
                                                                  source_id=ctx.artifacts.get("arxiv_id", "?"),
                                                                  locator=dk.key)])
            ctx.hops.append(HopRecord(index=len(ctx.hops), delegation_type="inline_resolve",
                                      cls=AdmissibilityClass.REORGANIZER, admitted=False,
                                      decision_key=dk.key, posterior=posterior,
                                      posterior_shift=shift))

    sections = compose_sections(ctx, dks)
    ungrounded, warnings = ground_and_graycheck(ctx, sections)
    flagged = flag_for_review(ctx.beliefs, dks)
    return BriefResult(
        title=f"Technical briefing: {ctx.artifacts.get('title', 'Untitled')}",
        paper_title=ctx.artifacts.get("title"),
        arxiv_id=ctx.artifacts.get("arxiv_id"),
        sections=sections,
        ungrounded_claims=ungrounded,
        flagged_for_review=flagged,
        warnings=warnings,
    )


def _orchestrator_node(state: _State) -> _State:
    """Emit the next Delegate (route to executor) or Finish (route to END)."""
    ctx = state["ctx"]
    econ = state["econ"]
    state["current_step"] = None
    state["current_cls"] = None

    # Expand the plan with verify + compose steps once metadata is available.
    if ctx.artifacts.get("title") and not state.get("verify_expanded"):
        dks = metric_keys_from_context(ctx)
        state["dks"] = dks
        for dk in dks:
            if ctx.tool_available("factcheck_metric"):
                step = PlanStep("verify", "verify_metric", ["factcheck_metric"],
                                "paper-text", f"Verify the figure {dk.key.split('::')[1]}.",
                                decision_key=dk.key, dk=dk)
                state["plan"].append(step)
        # Composition is a reorganizer step; the gate will collapse it.
        state["plan"].append(PlanStep("compose", "compose", ["draft_text"], "none",
                                      "Compose the briefing from gathered evidence."))
        state["verify_expanded"] = True

    # Process the plan, collapsing non-admitted steps inline until we either find
    # an admissible delegation (yield to executor) or run out of steps (finish).
    while state["plan"]:
        if state["steps_taken"] >= _MAX_STEPS:
            break
        step = state["plan"].pop(0)
        state["steps_taken"] += 1
        delegation = build_delegation(ctx, step)
        decision = gate_decision(delegation, REGISTRY, econ)
        state["gate_log"].append({
            "delegation_type": step.delegation_type,
            "cls": decision.cls.value,
            "admit": decision.admit,
            "redundant": bool(decision.redundant),
            "reason": decision.reason,
        })
        log_event(log, "gate decision", stage="orchestrate", agent="orchestrator",
                  delegation_class=decision.cls.value,
                  status="admit" if decision.admit else "reject")

        if decision.admit:
            state["current_step"] = step
            state["current_cls"] = decision.cls
            return state

        # Collapsed / rejected work stays with the orchestrator.
        if step.phase == "compose":
            state["result"] = _finalize(state)
        # (redundant exogenous steps are simply pruned.)

    if state.get("result") is None:
        state["result"] = _finalize(state)
    return state


def _executor_node(state: _State) -> _State:
    step = state["current_step"]
    cls = state["current_cls"]
    assert step is not None and cls is not None
    run_delegation(state["ctx"], step, cls, state["econ"])
    state["current_step"] = None
    state["current_cls"] = None
    return state


def _route(state: _State) -> str:
    return "executor" if state.get("current_step") is not None else END


def _build_graph():
    builder = StateGraph(_State)
    builder.add_node("orchestrator", _orchestrator_node)
    builder.add_node("executor", _executor_node)
    builder.set_entry_point("orchestrator")
    builder.add_conditional_edges("orchestrator", _route,
                                  {"executor": "executor", END: END})
    builder.add_edge("executor", "orchestrator")
    return builder.compile()


class GovernedOrchestrator:
    """Runs a briefing request through the admissibility-gated LangGraph."""

    def __init__(self) -> None:
        self._graph = _build_graph()

    def run(self, request: BriefRequest, *, disabled_tools: set[str] | None = None) -> GovernedResult:
        ctx = new_context(request, disabled_tools)
        if request.arxiv_id:
            ctx.artifacts["arxiv_id"] = request.arxiv_id
        state: _State = {
            "ctx": ctx,
            "econ": DelegationEconomics(),
            "plan": _initial_plan(ctx),
            "dks": [],
            "current_step": None,
            "current_cls": None,
            "verify_expanded": False,
            "gate_log": [],
            "result": None,
            "steps_taken": 0,
        }
        final = self._graph.invoke(state, config={"recursion_limit": 200})
        result = final["result"]
        return GovernedResult(
            brief=result,
            ctx=final["ctx"],
            econ=final["econ"],
            gate_log=final["gate_log"],
            hops=final["ctx"].hops,
        )
