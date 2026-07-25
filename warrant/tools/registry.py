"""Tool registry with the exogenous/validator/transform tagging.

The tag is the *structural* half of the admissibility gate: a worker equipped
only with TRANSFORM tools (or none) cannot inject a new exogenous signal, so it
is a Reorganizer by construction, regardless of what its prompt claims.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field

from warrant.exceptions import ToolError
from warrant.schemas.belief import EvidenceRef


class ToolRole(str, Enum):
    """What kind of signal a tool can contribute."""

    INJECTOR = "INJECTOR"    # brings new external information (arxiv, pdf, web, youtube)
    VALIDATOR = "VALIDATOR"  # runs an external/executable non-redundant check
    TRANSFORM = "TRANSFORM"  # pure reorganization of existing context (no new signal)


class ToolResult(BaseModel):
    observation: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    name: str
    role: ToolRole
    signal_source: str = Field(..., description="e.g. 'arxiv-db', 'paper-text', 'none'")
    description: str = ""
    run: Callable[..., ToolResult]

    model_config = {"arbitrary_types_allowed": True}

    @property
    def exogenous(self) -> bool:
        """True iff this tool can move the Bayes envelope (injector or validator)."""
        return self.role in (ToolRole.INJECTOR, ToolRole.VALIDATOR)


class ToolRegistry:
    """A namespace of registered tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ToolError(spec.name, "tool already registered")
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(name, "tool not registered") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, names: list[str]) -> list[ToolSpec]:
        return [self.get(n) for n in names]


# Process-wide default registry, populated on import of `warrant.tools`.
REGISTRY = ToolRegistry()


def register(
    name: str,
    role: ToolRole,
    signal_source: str,
    description: str = "",
) -> Callable[[Callable[..., ToolResult]], Callable[..., ToolResult]]:
    """Decorator registering a function as a tool in the default registry."""

    def deco(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        REGISTRY.register(
            ToolSpec(
                name=name,
                role=role,
                signal_source=signal_source,
                description=description or (fn.__doc__ or "").strip(),
                run=fn,
            )
        )
        return fn

    return deco
