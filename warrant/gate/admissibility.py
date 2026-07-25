"""The admissibility gate.

Two layers:

1. **Structural (a-priori):** inspect the delegation's tools. A worker with no
   exogenous tool cannot inject signal and is a Reorganizer by construction —
   no prompt can override that. Injector tools win over validator tools.
2. **Economic (a-posteriori):** consult the :class:`DelegationEconomics` ledger;
   a delegation *type* that has historically failed to move the posterior is
   pruned even if it is structurally exogenous.

Admitted classes: INJECTOR, VALIDATOR. Rejected/collapsed: REORGANIZER (and any
type flagged redundant by the economics ledger).
"""

from __future__ import annotations

from warrant.gate.novelty_audit import DelegationEconomics
from warrant.schemas.belief import AdmissibilityClass, Delegation, GateDecision
from warrant.tools.registry import ToolRegistry, ToolRole


def classify_delegation(delegation: Delegation, registry: ToolRegistry) -> AdmissibilityClass:
    """Structural classification from the delegation's tool tags."""
    roles = [registry.get(name).role for name in delegation.tools if registry.has(name)]
    if ToolRole.INJECTOR in roles:
        return AdmissibilityClass.INJECTOR
    if ToolRole.VALIDATOR in roles:
        return AdmissibilityClass.VALIDATOR
    return AdmissibilityClass.REORGANIZER


def gate_decision(
    delegation: Delegation,
    registry: ToolRegistry,
    economics: DelegationEconomics | None = None,
) -> GateDecision:
    """Return the gate's verdict for a proposed delegation."""
    cls = classify_delegation(delegation, registry)

    if cls is AdmissibilityClass.REORGANIZER:
        return GateDecision(
            admit=False,
            cls=cls,
            reason=(
                "REORGANIZER: worker has no exogenous tool; it would only "
                "re-process shared context. Dominated by a single centralized "
                "call — collapsing into the orchestrator."
            ),
        )

    if economics is not None and economics.is_redundant_type(delegation.delegation_type):
        return GateDecision(
            admit=False,
            cls=cls,
            redundant=True,
            reason=(
                f"REDUNDANT: delegation type {delegation.delegation_type!r} has a "
                f"mean posterior shift of {economics.mean_shift(delegation.delegation_type):.4f} "
                f"bits (< epsilon). Its exogenous tool returns information already "
                f"in the belief state — pruning."
            ),
        )

    signal = delegation.signal_claim.expected_signal_source or "external source"
    return GateDecision(
        admit=True,
        cls=cls,
        reason=f"{cls.value}: injects new signal from {signal}.",
    )
