"""Structural gate audit (capability 1) — zero-config reorganizer detection.

Aggregates a node's behaviour across every recorded run and classifies it with
the *same* rule the runtime gate uses (``gate/admissibility.py``): a node that
never calls an exogenous (injector/validator) tool cannot inject a new signal
and is a REORGANIZER by construction — a candidate to collapse into the
orchestrator. This needs only tool tags; no annotation, no re-runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from warrant.schemas.belief import AdmissibilityClass
from warrant.tools.registry import ToolRole
from warrant.trace.contract import RunTrace


@dataclass
class NodeStructure:
    """Aggregated structural facts about one node across all runs."""

    node_id: str
    runs: int = 0
    exogenous_runs: int = 0
    injector_runs: int = 0
    validator_runs: int = 0
    tool_names: set[str] = field(default_factory=set)

    @property
    def admissibility(self) -> AdmissibilityClass:
        # A node counts as exogenous if it ever injected signal; reorganizer only
        # if it *never* did across observed runs. Injecting outranks validating,
        # matching ``gate/admissibility.py``'s precedence.
        if self.injector_runs:
            return AdmissibilityClass.INJECTOR
        if self.validator_runs:
            return AdmissibilityClass.VALIDATOR
        return AdmissibilityClass.REORGANIZER

    @property
    def is_reorganizer(self) -> bool:
        return self.admissibility is AdmissibilityClass.REORGANIZER


def structural_audit(traces: list[RunTrace]) -> dict[str, NodeStructure]:
    """Classify every node across ``traces`` by exogeneity.

    Preserves first-seen order of nodes so downstream reports read like the graph.
    """
    out: dict[str, NodeStructure] = {}
    for trace in traces:
        for node in trace.nodes:
            struct = out.setdefault(node.node_id, NodeStructure(node_id=node.node_id))
            struct.runs += 1
            roles = {tc.role for tc in node.tool_calls if tc.exogenous}
            if node.has_exogenous_tool:
                struct.exogenous_runs += 1
            if ToolRole.INJECTOR in roles:
                struct.injector_runs += 1
            elif ToolRole.VALIDATOR in roles:
                struct.validator_runs += 1
            struct.tool_names.update(node.tool_names)
    return out


__all__ = ["NodeStructure", "structural_audit"]
