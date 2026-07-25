"""Anthropic adapter — used for the orchestrator (planner / approval gate)."""

from __future__ import annotations

from warrant.config import Settings, get_settings
from warrant.exceptions import ProviderError
from warrant.providers.base import (
    CompletionResult,
    average_posteriors,
    build_posterior_prompt,
    parse_distribution,
)
from warrant.schemas.belief import Posterior


class AnthropicProvider:
    """Thin wrapper over the Anthropic Messages API.

    Kept deliberately small: the rest of the codebase depends only on
    :class:`~warrant.providers.base.LLMProvider`, never on the SDK.
    """

    name = "anthropic"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.require_live()
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        except Exception as exc:  # pragma: no cover - exercised only live
            raise ProviderError("anthropic", "failed to initialize client", cause=exc)

    def _default_model(self, model: str | None) -> str:
        return model or self.settings.orchestrator_model

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        try:
            resp = self._client.messages.create(
                model=self._default_model(model),
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in resp.content if block.type == "text")
            tokens = resp.usage.input_tokens + resp.usage.output_tokens
            return CompletionResult(text=text, tokens=tokens, model=self._default_model(model))
        except Exception as exc:  # pragma: no cover - exercised only live
            raise ProviderError("anthropic", "completion failed", cause=exc)

    def classify_posterior(
        self,
        *,
        question: str,
        options: list[str],
        context: str,
        model: str | None = None,
        k: int | None = None,
        hints: dict[str, list[str]] | None = None,
    ) -> Posterior:
        k = k or self.settings.posterior_samples
        prompt = build_posterior_prompt(question, options, context)
        samples: list[dict[str, float]] = []
        for _ in range(k):
            result = self.complete(
                system="Return only a JSON probability distribution.",
                prompt=prompt,
                model=model,
                temperature=self.settings.posterior_temperature,
                max_tokens=256,
            )
            samples.append(parse_distribution(result.text, options))
        return average_posteriors(samples, options)
