"""Embedding-/text-novelty economics (capability 3).

Generalizes the research engine's ``DelegationEconomics`` from a KL posterior
shift to a *pluggable distance*: how novel is a node's output relative to the
context it already had (the outputs of the nodes that ran before it)? A node
whose output is largely re-statement of prior context has low novelty — a
model-free proxy for "did not move the posterior", so a reorganizer candidate.

The default distance is offline and deterministic (Jaccard distance over word
shingles); an embeddings-backed cosine distance can be injected when a key is
available, but is never required — offline determinism is mandatory
(PRODUCT_DIRECTION.md §6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from warrant.gate.novelty_audit import DelegationEconomics
from warrant.trace.contract import RunTrace

Distance = Callable[[str, str], float]

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def jaccard_distance(output: str, context: str) -> float:
    """1 - |A∩B|/|A∪B| over word sets. 0 = fully redundant, 1 = fully novel.

    Measured on the output's *own* tokens so that a longer context cannot dilute
    a genuinely novel output: novelty = fraction of the output's tokens absent
    from the context.
    """
    out = _tokens(output)
    if not out:
        return 0.0
    ctx = _tokens(context)
    new = out - ctx
    return len(new) / len(out)


@dataclass
class NodeNovelty:
    node_id: str
    runs: int = 0
    total_novelty: float = 0.0

    @property
    def mean_novelty(self) -> float:
        return self.total_novelty / self.runs if self.runs else 0.0


def novelty_audit(
    traces: list[RunTrace],
    distance: Distance = jaccard_distance,
    epsilon: float = 0.15,
) -> dict[str, NodeNovelty]:
    """Mean novelty of each node's output vs its prior in-run context.

    ``epsilon`` mirrors ``DelegationEconomics``' pruning threshold: nodes whose
    mean novelty falls below it are flagged redundant by the caller. The running
    per-node economics ledger is reused so the SDK and the research engine share
    one notion of "paying for itself in information".
    """
    econ = DelegationEconomics(epsilon=epsilon, min_observations=1)
    out: dict[str, NodeNovelty] = {}
    for trace in traces:
        prior = ""
        for node in trace.nodes:
            nov = distance(node.outcome.output_text, prior)
            rec = out.setdefault(node.node_id, NodeNovelty(node_id=node.node_id))
            rec.runs += 1
            rec.total_novelty += nov
            econ.record(node.node_id, nov)
            prior += "\n" + node.outcome.output_text
    return out


def is_redundant(novelty: NodeNovelty, epsilon: float = 0.15) -> bool:
    return novelty.mean_novelty < epsilon


__all__ = ["NodeNovelty", "novelty_audit", "jaccard_distance", "is_redundant", "Distance"]
