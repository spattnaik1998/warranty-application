"""Text-novelty economics (capability 3).

Generalizes the research engine's ``DelegationEconomics`` from a KL posterior
shift to a *pluggable distance*: how novel is a node's output relative to the
context it already had (the outputs of the nodes that ran before it)? A node
whose output is largely re-statement of prior context has low novelty — a
model-free proxy for "did not move the posterior", so a reorganizer candidate.

The default distance is offline, deterministic, and lexical; an embeddings-backed
cosine distance can be injected through the ``distance`` parameter, but is never
required — offline determinism is mandatory (PRODUCT_DIRECTION.md §6). This is a
*proxy*, not a measurement: it sees vocabulary, not meaning, so a node that
rephrases its context scores as novel. Ablation is what settles the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from warrant.trace.contract import RunTrace

Distance = Callable[[str, str], float]

_WORD = re.compile(r"[a-z0-9]+")

_DEFAULT_EPSILON = 0.15


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def unseen_token_fraction(output: str, context: str) -> float:
    """Fraction of the output's word types absent from the context.

    ``|A \\ B| / |A|`` — deliberately asymmetric rather than Jaccard, so that a
    long accumulated context cannot dilute a genuinely novel output. 0 = every
    word already appeared in context (pure restatement), 1 = all new vocabulary.
    """
    out = _tokens(output)
    if not out:
        return 0.0
    return len(out - _tokens(context)) / len(out)


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
    distance: Distance = unseen_token_fraction,
    epsilon: float = _DEFAULT_EPSILON,
) -> dict[str, NodeNovelty]:
    """Mean novelty of each node's output vs its prior in-run context.

    ``epsilon`` is the caller's redundancy threshold, carried here so callers can
    ask ``is_redundant`` with the same number the audit used.

    Note the first node of a run always scores 1.0: its context is empty, so every
    token is "new". That is a structural artifact of position, not evidence, and
    the audit treats an entry node's novelty accordingly.
    """
    out: dict[str, NodeNovelty] = {}
    for trace in traces:
        prior = ""
        for node in trace.nodes:
            nov = distance(node.outcome.output_text, prior)
            rec = out.setdefault(node.node_id, NodeNovelty(node_id=node.node_id))
            rec.runs += 1
            rec.total_novelty += nov
            prior += "\n" + node.outcome.output_text
    return out


def is_redundant(novelty: NodeNovelty, epsilon: float = _DEFAULT_EPSILON) -> bool:
    return novelty.mean_novelty < epsilon


__all__ = ["NodeNovelty", "novelty_audit", "unseen_token_fraction", "is_redundant", "Distance"]
