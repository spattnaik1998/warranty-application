"""The Delegation Ledger probe.

Runs a briefing task under matched conditions and reproduces, on our own
pipeline, the paper's core findings:

* **A (centralized)** — one decision-maker with all evidence: the upper bound.
* **B (governed)** — the admissibility-gated network: should match A.
* **B- (naive)** — gate disabled, prose relay of increasing depth: the
  telephone-game collapse.
* **C (signal-starved)** — a genuine retrieval tool removed: accuracy drops
  because the signal never enters, no matter how many agents talk.

It also sweeps relay depth × interface to plot accuracy-vs-depth and the
KL-vs-accuracy-drop correlation (the paper's r≈0.72 result), and to contrast
the prose relay against the posterior-preserving interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from warrant.belief.distortion import pearson
from warrant.belief.state import BeliefState
from warrant.ledger.metrics import (
    accuracy,
    distortion_to_centralized,
    mean_kl_to_centralized,
    relayed_context,
)
from warrant.logging_setup import get_logger, log_event
from warrant.orchestrator import GovernedOrchestrator
from warrant.pipeline.steps import (
    DecisionKey,
    PipelineContext,
    decision_keys,
    discover_target,
    fetch_paper,
    new_context,
    read_paper,
)
from warrant.schemas.belief import Posterior
from warrant.schemas.ledger import Condition, RunRecord
from warrant.schemas.tasks import BriefRequest

log = get_logger("ledger")

MAX_DEPTH = 5


@dataclass
class SweepPoint:
    depth: int
    interface: str
    accuracy: float
    mean_kl: float
    accuracy_drop: float


@dataclass
class ProbeResult:
    task_id: str
    dks: list[DecisionKey]
    centralized: dict[str, Posterior]
    runs: dict[Condition, RunRecord] = field(default_factory=dict)
    sweep: list[SweepPoint] = field(default_factory=list)
    kl_accuracy_r: float = 0.0

    def prose_slope(self) -> float:
        """Accuracy points lost per relay stage under the prose interface."""
        prose = [p for p in self.sweep if p.interface == "prose"]
        if len(prose) < 2:
            return 0.0
        first, last = prose[0], prose[-1]
        span = last.depth - first.depth
        return (first.accuracy - last.accuracy) * 100 / span if span else 0.0

    def posterior_slope(self) -> float:
        post = [p for p in self.sweep if p.interface == "posterior"]
        if len(post) < 2:
            return 0.0
        first, last = post[0], post[-1]
        span = last.depth - first.depth
        return (first.accuracy - last.accuracy) * 100 / span if span else 0.0


def _prepare(request: BriefRequest) -> tuple[PipelineContext, list[DecisionKey], str]:
    ctx = new_context(request)
    arxiv_id = discover_target(ctx)
    fetch_paper(ctx, arxiv_id)
    source_text = read_paper(ctx, arxiv_id)
    dks = decision_keys(ctx)
    full = "\n".join(str(v) for v in ctx.artifacts.values())
    return ctx, dks, source_text or full


def _elicit_from(ctx: PipelineContext, dk: DecisionKey, context: str, beliefs: BeliefState) -> None:
    posterior = ctx.worker.classify_posterior(
        question=dk.claim, options=dk.options, context=context, hints=dk.hints)
    beliefs.update(dk.key, posterior, claim=dk.claim)


def _run_centralized(request: BriefRequest, dks, source_text) -> tuple[RunRecord, dict[str, Posterior]]:
    ctx = new_context(request)
    beliefs = BeliefState()
    for dk in dks:
        _elicit_from(ctx, dk, source_text, beliefs)  # full evidence, one decision-maker
    rec = RunRecord(
        task_id="brief", condition=Condition.CENTRALIZED,
        accuracy=accuracy(beliefs, dks), delegation_depth=1,
        admitted_delegations=0, communication_loss=0.0,
    )
    return rec, beliefs.snapshot()


def _run_governed(request: BriefRequest, dks, centralized) -> RunRecord:
    res = GovernedOrchestrator().run(request)
    return RunRecord(
        task_id="brief", condition=Condition.GOVERNED,
        accuracy=accuracy(res.ctx.beliefs, dks),
        communication_loss=distortion_to_centralized(res.ctx.beliefs, centralized, dks),
        delegation_depth=res.admitted,
        admitted_delegations=res.admitted,
        rejected_delegations=res.rejected,
        redundant_delegations=res.redundant,
        total_tokens=sum(h.tokens for h in res.hops),
    )


def _run_naive(request, dks, source_text, centralized, depth: int, interface: str = "prose") -> RunRecord:
    ctx = new_context(request)
    beliefs = BeliefState()
    for dk in dks:
        context = relayed_context(dk, source_text, depth, interface)
        _elicit_from(ctx, dk, context, beliefs)
    return RunRecord(
        task_id="brief", condition=Condition.NAIVE,
        accuracy=accuracy(beliefs, dks),
        communication_loss=distortion_to_centralized(beliefs, centralized, dks),
        delegation_depth=depth,
    )


def _run_signal_starved(request: BriefRequest, dks, centralized) -> RunRecord:
    res = GovernedOrchestrator().run(request, disabled_tools={"pdf_read", "factcheck_metric"})
    return RunRecord(
        task_id="brief", condition=Condition.SIGNAL_STARVED,
        accuracy=accuracy(res.ctx.beliefs, dks),
        communication_loss=distortion_to_centralized(res.ctx.beliefs, centralized, dks),
        delegation_depth=res.admitted,
        admitted_delegations=res.admitted,
        rejected_delegations=res.rejected,
    )


def run_probe(request: BriefRequest | None = None) -> ProbeResult:
    """Execute all conditions plus the depth×interface sweep."""
    request = request or BriefRequest(youtube_channel="Last Week in AI")
    _, dks, source_text = _prepare(request)

    centralized_rec, centralized = _run_centralized(request, dks, source_text)
    result = ProbeResult(task_id="brief", dks=dks, centralized=centralized)
    result.runs[Condition.CENTRALIZED] = centralized_rec
    result.runs[Condition.GOVERNED] = _run_governed(request, dks, centralized)
    result.runs[Condition.NAIVE] = _run_naive(request, dks, source_text, centralized,
                                              depth=MAX_DEPTH, interface="prose")
    result.runs[Condition.SIGNAL_STARVED] = _run_signal_starved(request, dks, centralized)

    # Depth × interface sweep.
    centralized_acc = centralized_rec.accuracy
    kls: list[float] = []
    drops: list[float] = []
    for interface in ("prose", "posterior"):
        for depth in range(0, MAX_DEPTH + 1):
            ctx = new_context(request)
            beliefs = BeliefState()
            for dk in dks:
                _elicit_from(ctx, dk, relayed_context(dk, source_text, depth, interface), beliefs)
            acc = accuracy(beliefs, dks)
            mkl = mean_kl_to_centralized(beliefs, centralized, dks)
            drop = centralized_acc - acc
            result.sweep.append(SweepPoint(depth, interface, acc, mkl, drop))
            if interface == "prose":
                kls.append(mkl)
                drops.append(drop)
    result.kl_accuracy_r = pearson(kls, drops)

    log_event(log, "probe complete", stage="ledger", status="ok",
              distortion=round(result.runs[Condition.NAIVE].communication_loss, 4))
    return result
