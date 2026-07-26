"""Posterior-distortion analysis (capability 4) — the rigorous, theorem-grade proof.

Active only when the developer annotated decision points via
``warrant.decision(...)``. It reuses the research engine's distortion metrics
(``belief/distortion.py``) to measure the "telephone game" directly: how much
each relay hop garbles the shared posterior. When fewer than two decision points
exist it degrades gracefully to ``available=False`` and the audit relies on the
structural + ablation + novelty signals instead — never fabricating a distortion
number it cannot measure (PRODUCT_DIRECTION.md §6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from warrant.belief.distortion import serial_chain_loss
from warrant.trace.contract import RunTrace


@dataclass
class NodeDistortion:
    node_id: str
    hops: int = 0
    total_kl: float = 0.0

    @property
    def mean_kl(self) -> float:
        return self.total_kl / self.hops if self.hops else 0.0


@dataclass
class DistortionReport:
    available: bool = False
    mean_chain_loss_bits: float = 0.0
    per_node: dict[str, NodeDistortion] = field(default_factory=dict)


def distortion_audit(traces: list[RunTrace]) -> DistortionReport:
    """Measure additive posterior distortion along each run's decision chain."""
    chains: list[float] = []
    per_node: dict[str, NodeDistortion] = {}
    for trace in traces:
        annotated = [n for n in trace.nodes if n.decision and n.decision.posterior]
        if len(annotated) < 2:
            continue
        posteriors = [n.decision.posterior for n in annotated]
        chains.append(serial_chain_loss(posteriors))
        for i in range(1, len(annotated)):
            node = annotated[i]
            kl = posteriors[i - 1].kl_to(posteriors[i])
            rec = per_node.setdefault(node.node_id, NodeDistortion(node_id=node.node_id))
            rec.hops += 1
            rec.total_kl += kl
    if not chains:
        return DistortionReport(available=False)
    return DistortionReport(
        available=True,
        mean_chain_loss_bits=sum(chains) / len(chains),
        per_node=per_node,
    )


__all__ = ["NodeDistortion", "DistortionReport", "distortion_audit"]
