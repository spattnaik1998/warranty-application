"""Centralized, environment-driven settings with fail-fast validation.

All secrets and tunables live here so business logic never reads os.environ
directly and model names are never hardcoded deep in the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from warrant.exceptions import ConfigError

try:  # optional convenience; not required
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is best-effort
    pass


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float, got {raw!r}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an int, got {raw!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    # Credentials (may be empty in mock mode).
    anthropic_api_key: str
    openai_api_key: str

    # Models (configurable via env).
    orchestrator_model: str
    worker_model: str
    worker_strong_model: str

    # Run mode.
    mock: bool

    # Posterior elicitation.
    posterior_samples: int
    posterior_temperature: float

    # Delegation economics.
    novelty_kl_epsilon: float

    # Risk-triggered escalation (Theorem 10).
    wrong_cost: float
    review_cost: float

    # Output.
    output_dir: Path
    log_level: str

    @classmethod
    def load(cls) -> "Settings":
        """Build settings from the environment."""
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            orchestrator_model=os.getenv("WARRANT_ORCHESTRATOR_MODEL", "claude-opus-4-8"),
            worker_model=os.getenv("WARRANT_WORKER_MODEL", "gpt-4o-mini"),
            worker_strong_model=os.getenv("WARRANT_WORKER_STRONG_MODEL", "gpt-4o"),
            mock=_get_bool("WARRANT_MOCK", True),
            posterior_samples=_get_int("WARRANT_POSTERIOR_SAMPLES", 5),
            posterior_temperature=_get_float("WARRANT_POSTERIOR_TEMPERATURE", 0.7),
            novelty_kl_epsilon=_get_float("WARRANT_NOVELTY_KL_EPSILON", 0.05),
            wrong_cost=_get_float("WARRANT_WRONG_COST", 1.0),
            review_cost=_get_float("WARRANT_REVIEW_COST", 0.15),
            output_dir=Path(os.getenv("WARRANT_OUTPUT_DIR", "./out")).resolve(),
            log_level=os.getenv("WARRANT_LOG_LEVEL", "INFO"),
        )

    def require_live(self) -> None:
        """Fail fast if live mode is requested without the needed credentials."""
        if self.mock:
            return
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if missing:
            raise ConfigError(
                "Live mode requires credentials but these are missing: "
                + ", ".join(missing)
                + ". Set them in your environment or run with WARRANT_MOCK=1."
            )

    def validate(self) -> "Settings":
        """Validate tunable ranges; return self for chaining."""
        if self.posterior_samples < 1:
            raise ConfigError("WARRANT_POSTERIOR_SAMPLES must be >= 1")
        if not (0.0 <= self.posterior_temperature <= 2.0):
            raise ConfigError("WARRANT_POSTERIOR_TEMPERATURE must be in [0, 2]")
        if self.novelty_kl_epsilon < 0:
            raise ConfigError("WARRANT_NOVELTY_KL_EPSILON must be >= 0")
        if self.review_cost < 0 or self.wrong_cost < 0:
            raise ConfigError("cost parameters must be >= 0")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide validated settings singleton."""
    return Settings.load().validate()
