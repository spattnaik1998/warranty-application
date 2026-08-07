"""The normalized, framework-agnostic run trace.

A run of *any* multi-agent system is reduced to::

    RunTrace
      └── NodeRun            (one agent / graph node execution)
            ├── ToolCallRecord*   (each tool the node invoked, tagged by role)
            ├── DecisionPoint?    (optional posterior annotation, level-4)
            └── Outcome           (cost, latency, tokens, output digest)

This is the only structure the analysis layer sees. Framework adapters
(``warrant.adapters.*``) are responsible for producing it; they may import
LangGraph, but nothing downstream of the contract may. Reusing the research
engine's own vocabulary (``ToolRole``, ``AdmissibilityClass``, ``Posterior``)
keeps the SDK and the theorem in one type system.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from warrant.schemas.belief import AdmissibilityClass, Posterior
from warrant.tools.registry import ToolRole


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def content_digest(text: str) -> str:
    """Stable content address for an output, used by ablation diffing."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class RunStatus(str, Enum):
    """Terminal status of a run."""

    OK = "OK"
    ERROR = "ERROR"
    PARTIAL = "PARTIAL"


class ToolCallRecord(BaseModel):
    """One tool invocation made by a node.

    ``role`` is the exogeneity tag that the structural audit keys on: a node
    that only ever calls ``TRANSFORM`` tools (or no tools) cannot inject a new
    exogenous signal and is a reorganizer candidate. ``role`` may be ``None``
    when the adapter could not classify the tool; the auto-tagging heuristic
    or an explicit ``warrant.tool_tag`` fills it in.
    """

    name: str
    role: ToolRole | None = None
    signal_source: str | None = None
    tokens: int = 0
    latency_ms: int = 0
    ok: bool = True
    error: str | None = None

    @property
    def exogenous(self) -> bool:
        """True iff this call could move the Bayes envelope (injector/validator)."""
        return self.role in (ToolRole.INJECTOR, ToolRole.VALIDATOR)


class DecisionPoint(BaseModel):
    """An optional, user-annotated belief checkpoint (capability level 4).

    Present only when the developer calls ``warrant.decision(key, options,
    posterior)``. When present, it unlocks the rigorous posterior-distortion
    analysis and the matched-condition ledger; when absent, the audit falls
    back to structural + ablation + novelty signals.
    """

    key: str = Field(..., description="belief key, e.g. 'headline_metric_matches'")
    options: list[str] = Field(default_factory=list, description="label space")
    posterior: Posterior | None = None
    claim: str = ""


class Outcome(BaseModel):
    """The measurable result of a node run: what it cost and what it produced."""

    status: RunStatus = RunStatus.OK
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    output_text: str = ""
    output_digest: str = ""
    error: str | None = None
    # Provenance of ``tokens`` so a dollar figure never overstates its certainty:
    #   "measured"  — real usage reported by the model (billing-accurate)
    #   "estimated" — node ran a model but reported no usage; ~4 chars/token proxy
    #   "none"      — node made no billable model call (e.g. pure retrieval); cost $0
    token_source: str = "estimated"
    # Which model actually billed these tokens, when the framework reports it.
    # ``None`` means the adapter could not tell, and the audit must fall back to a
    # configured default — a fact the report discloses rather than hides.
    model: str | None = None
    # True when one node's messages came from more than one model; ``model`` then
    # names the one that accounted for the most tokens.
    mixed_models: bool = False

    def finalize(self) -> "Outcome":
        """Fill the output digest from the text if not already set."""
        if self.output_text and not self.output_digest:
            self.output_digest = content_digest(self.output_text)
        return self


class NodeRun(BaseModel):
    """One agent / graph-node execution within a run.

    A node is a *delegation* in the theorem's sense: the orchestrator handed it
    an instruction and context and expects a signal back. ``admissibility`` is
    left ``None`` by adapters and filled by the structural analyzer.
    """

    node_id: str = Field(..., description="stable node/agent name in the graph")
    step: int = Field(0, description="execution order within the run")
    instruction: str = ""
    context_keys: list[str] = Field(
        default_factory=list, description="context items the node was given"
    )
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    decision: DecisionPoint | None = None
    outcome: Outcome = Field(default_factory=Outcome)
    admissibility: AdmissibilityClass | None = None
    started_at: datetime = Field(default_factory=_utcnow)

    @property
    def has_exogenous_tool(self) -> bool:
        """True iff the node called at least one injector/validator tool."""
        return any(tc.exogenous for tc in self.tool_calls)

    @property
    def tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]


class RunTrace(BaseModel):
    """A single end-to-end run of an instrumented multi-agent system."""

    run_id: str
    graph_name: str = ""
    mode: str = "observe"
    status: RunStatus = RunStatus.OK
    nodes: list[NodeRun] = Field(default_factory=list)
    final_output: str = ""
    final_digest: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)

    def add_node(self, node: NodeRun) -> NodeRun:
        node.outcome = node.outcome.finalize()
        if node.step == 0:
            node.step = len(self.nodes)
        self.nodes.append(node)
        return node

    def finalize(self, final_output: str | None = None) -> "RunTrace":
        if final_output is not None:
            self.final_output = final_output
        if self.final_output and not self.final_digest:
            self.final_digest = content_digest(self.final_output)
        return self

    def node(self, node_id: str) -> NodeRun | None:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    @property
    def total_tokens(self) -> int:
        return sum(n.outcome.tokens for n in self.nodes)

    @property
    def total_latency_ms(self) -> int:
        return sum(n.outcome.latency_ms for n in self.nodes)


__all__ = [
    "RunTrace",
    "NodeRun",
    "ToolCallRecord",
    "DecisionPoint",
    "Outcome",
    "RunStatus",
    "content_digest",
]
