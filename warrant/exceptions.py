"""Domain-specific exceptions.

External provider and tool failures are wrapped in these so callers never see
raw SDK exceptions and error messages stay actionable (per project rules:
never swallow exceptions silently; return actionable messages).
"""

from __future__ import annotations


class WarrantError(Exception):
    """Base class for all Warrant domain errors."""


class ConfigError(WarrantError):
    """Raised when required configuration is missing or invalid (fail fast)."""


class ProviderError(WarrantError):
    """Wraps a failure from an LLM provider adapter (Anthropic / OpenAI)."""

    def __init__(self, provider: str, message: str, *, cause: Exception | None = None):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")
        if cause is not None:
            self.__cause__ = cause


class ToolError(WarrantError):
    """Wraps a failure from an external tool (arXiv, PDF read, web, ...)."""

    def __init__(self, tool: str, message: str, *, cause: Exception | None = None):
        self.tool = tool
        super().__init__(f"[tool:{tool}] {message}")
        if cause is not None:
            self.__cause__ = cause


class GateRejection(WarrantError):
    """Raised when the admissibility gate refuses a proposed delegation.

    Carries the structured reason so the orchestrator can log it and either
    collapse the work into a single call or re-specify the delegation.
    """

    def __init__(self, reason: str, *, delegation_class: str | None = None):
        self.delegation_class = delegation_class
        super().__init__(reason)


class PosteriorError(WarrantError):
    """Raised when a posterior distribution cannot be elicited or is invalid."""
