"""Reusable pipeline primitives shared by the naive runner and the orchestrator.

Each step is a small, testable unit. The exogenous steps (discover / fetch /
read / factcheck) inject signal; the composition step only reorganizes evidence
already gathered — which is exactly why, under governance, composition is done
by the orchestrator itself rather than delegated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from warrant.belief.state import BeliefState
from warrant.config import Settings, get_settings
from warrant.exceptions import ToolError
from warrant.logging_setup import get_logger, log_event
from warrant.providers import LLMProvider, get_provider
from warrant.schemas.belief import EvidenceRef, Posterior
from warrant.schemas.ledger import HopRecord
from warrant.schemas.tasks import BriefRequest, Claim, Section
from warrant.tools import REGISTRY

log = get_logger("pipeline")


@dataclass
class DecisionKey:
    """A load-bearing belief the terminal briefing depends on."""

    key: str
    claim: str
    options: list[str]
    correct: str
    hints: dict[str, list[str]]  # option -> evidence keywords


@dataclass
class PipelineContext:
    request: BriefRequest
    worker: LLMProvider
    beliefs: BeliefState = field(default_factory=BeliefState)
    artifacts: dict[str, str] = field(default_factory=dict)
    hops: list[HopRecord] = field(default_factory=list)
    settings: Settings = field(default_factory=get_settings)
    # Tools deliberately withheld (for the signal-starved ledger condition).
    disabled_tools: set[str] = field(default_factory=set)

    def tool_available(self, name: str) -> bool:
        return name not in self.disabled_tools

    def run_tool(self, name: str, **kwargs):
        if not self.tool_available(name):
            raise ToolError(name, "tool disabled for this run (signal-starved)")
        return REGISTRY.get(name).run(**kwargs)


def new_context(request: BriefRequest, disabled_tools: set[str] | None = None) -> PipelineContext:
    return PipelineContext(
        request=request,
        worker=get_provider("worker"),
        disabled_tools=disabled_tools or set(),
    )


# --------------------------------------------------------------------------- #
# Exogenous steps
# --------------------------------------------------------------------------- #
def discover_target(ctx: PipelineContext) -> str:
    """Resolve the target arXiv id from the request (exogenous discovery)."""
    req = ctx.request
    if req.arxiv_id:
        return req.arxiv_id
    if req.youtube_channel and ctx.tool_available("youtube_latest"):
        result = ctx.run_tool("youtube_latest", channel=req.youtube_channel)
        ids = result.artifacts.get("arxiv_ids", "").split(",")
        ctx.artifacts["video_url"] = result.artifacts.get("url", "")
        if ids and ids[0]:
            log_event(log, "discovered via youtube", stage="discover",
                      tool="youtube_latest", arxiv_id=ids[0], status="ok")
            return ids[0]
    if req.arxiv_query and ctx.tool_available("arxiv_search"):
        result = ctx.run_tool("arxiv_search", query=req.arxiv_query)
        return result.artifacts["arxiv_id"]
    raise ToolError("discover", "could not resolve a target arXiv id from the request")


def fetch_paper(ctx: PipelineContext, arxiv_id: str) -> None:
    """Fetch metadata (exogenous)."""
    result = ctx.run_tool("arxiv_fetch", arxiv_id=arxiv_id)
    ctx.artifacts["arxiv_id"] = arxiv_id
    ctx.artifacts["title"] = result.artifacts["title"]
    ctx.artifacts["authors"] = result.artifacts["authors"]
    ctx.artifacts["abstract"] = result.artifacts["abstract"]
    log_event(log, "fetched metadata", stage="fetch", tool="arxiv_fetch",
              arxiv_id=arxiv_id, status="ok")


def read_paper(ctx: PipelineContext, arxiv_id: str) -> str:
    """Read the paper's full text (exogenous). Returns '' if the tool is disabled."""
    if not ctx.tool_available("pdf_read"):
        ctx.artifacts["paper_text"] = ""
        log_event(log, "pdf_read disabled (signal-starved)", stage="read",
                  arxiv_id=arxiv_id, status="skipped")
        return ""
    result = ctx.run_tool("pdf_read", arxiv_id=arxiv_id)
    text = result.artifacts["text"]
    ctx.artifacts["paper_text"] = text
    log_event(log, "read paper text", stage="read", tool="pdf_read",
              arxiv_id=arxiv_id, status="ok", duration_ms=0)
    return text


def decision_keys(ctx: PipelineContext, max_metrics: int = 5) -> list[DecisionKey]:
    """Derive the load-bearing beliefs for this paper from fixtures/metadata."""
    from warrant.tools.fixtures import PAPERS

    arxiv_id = ctx.artifacts.get("arxiv_id", "")
    keys: list[DecisionKey] = []
    title = ctx.artifacts.get("title", "")
    # Belief 1: did we identify the right paper?
    title_token = title.split()[0].lower() if title else "paper"
    keys.append(DecisionKey(
        key="paper_identified",
        claim=f"The briefing is about the paper titled '{title}'.",
        options=["correct", "wrong"],
        correct="correct",
        hints={"correct": [title_token, arxiv_id], "wrong": []},
    ))
    # Belief 2..N: is each headline metric supported by the source?
    metrics = PAPERS.get(arxiv_id, {}).get("key_metrics", [])[:max_metrics]
    for m in metrics:
        keys.append(DecisionKey(
            key=f"metric::{m}",
            claim=f"The source reports the figure {m}.",
            options=["supported", "unsupported"],
            correct="supported",
            hints={"supported": [m], "unsupported": []},
        ))
    return keys


def elicit_key(
    ctx: PipelineContext,
    dk: DecisionKey,
    context_text: str,
    *,
    delegation_type: str,
    cls,
    admitted: bool = True,
) -> Posterior:
    """Elicit a posterior for a decision key and fold it into the belief state."""
    prior = ctx.beliefs.posterior(dk.key)
    posterior = ctx.worker.classify_posterior(
        question=dk.claim,
        options=dk.options,
        context=context_text,
        hints=dk.hints,
    )
    shift = ctx.beliefs.update(
        dk.key, posterior, claim=dk.claim,
        evidence_refs=[EvidenceRef(source_type="pdf",
                                   source_id=ctx.artifacts.get("arxiv_id", "?"),
                                   locator=dk.key)],
    )
    ctx.hops.append(HopRecord(
        index=len(ctx.hops),
        delegation_type=delegation_type,
        cls=cls,
        admitted=admitted,
        decision_key=dk.key,
        prior=prior,
        posterior=posterior,
        posterior_shift=shift,
    ))
    log_event(log, "belief updated", stage="elicit", agent="worker",
              delegation_class=getattr(cls, "value", str(cls)),
              posterior_shift=round(shift, 4), status="ok")
    return posterior


# --------------------------------------------------------------------------- #
# Reorganization step (composition) — no new signal
# --------------------------------------------------------------------------- #
def compose_sections(ctx: PipelineContext, dks: list[DecisionKey]) -> list[Section]:
    """Compose the briefing from gathered evidence. Pure reorganization."""
    title = ctx.artifacts.get("title", "Untitled")
    authors = ctx.artifacts.get("authors", "")
    abstract = ctx.artifacts.get("abstract", "")
    arxiv_id = ctx.artifacts.get("arxiv_id", "")
    text = ctx.artifacts.get("paper_text", "")

    def ev(locator: str, snippet: str = "") -> list[EvidenceRef]:
        return [EvidenceRef(source_type="pdf", source_id=arxiv_id,
                            locator=locator, snippet=snippet[:120])]

    sections: list[Section] = []

    sections.append(Section(
        heading="Executive summary",
        body=(f"This briefing covers **{title}** by {authors}. {abstract}"),
        claims=[Claim(text=abstract, evidence_refs=ev("abstract", abstract),
                      confidence=ctx.beliefs.get("paper_identified").confidence
                      if ctx.beliefs.get("paper_identified") else 0.0)],
    ))

    sections.append(Section(
        heading="Paper identified",
        body=f"Title: {title}\nAuthors: {authors}\narXiv: {arxiv_id}",
        claims=[Claim(text=f"The paper is '{title}' ({arxiv_id}).",
                      evidence_refs=[EvidenceRef(source_type="arxiv", source_id=arxiv_id)],
                      confidence=1.0)],
    ))

    # Deep dive: quote each verified metric as a grounded claim.
    metric_claims: list[Claim] = []
    for dk in dks:
        if not dk.key.startswith("metric::"):
            continue
        metric = dk.key.split("::", 1)[1]
        post = ctx.beliefs.posterior(dk.key)
        conf = post.top_prob() if post else 0.0
        supported = bool(post and post.top_label() == "supported")
        refs = ev("metric", metric) if supported else []
        metric_claims.append(Claim(
            text=f"The paper reports {metric}.",
            evidence_refs=refs,
            confidence=conf,
        ))
    sections.append(Section(
        heading="Deep dive: methods and reported results",
        body=(text[:600] + ("…" if len(text) > 600 else "")) if text
        else "(source text unavailable for this run)",
        claims=metric_claims,
    ))

    sections.append(Section(
        heading="Practical implications for ML engineers",
        body=("Treat added agent roles as costly: a delegation is justified only "
              "when it injects a new exogenous signal or performs a non-redundant "
              "external check. Otherwise a single well-prompted call dominates."),
        claims=[Claim(text="Delegations without new exogenous signal are dominated "
                           "by a centralized decision-maker.",
                      evidence_refs=ev("theorem"), confidence=1.0)],
    ))

    sections.append(Section(
        heading="Open questions and limitations",
        body=("How much exogenous signal justifies an added stage remains an open "
              "optimization problem; results here are on a single paper corpus."),
        claims=[],
    ))
    return sections
