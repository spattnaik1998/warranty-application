"""OpenAI adapter — used for the worker executors (extraction / writing).

Posterior elicitation prefers token logprobs when the option set is a small
set of single tokens; otherwise it falls back to K-sample JSON averaging.
"""

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


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.require_live()
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.openai_api_key)
        except Exception as exc:  # pragma: no cover - exercised only live
            raise ProviderError("openai", "failed to initialize client", cause=exc)

    def _default_model(self, model: str | None) -> str:
        return model or self.settings.worker_model

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
            resp = self._client.chat.completions.create(
                model=self._default_model(model),
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else len(text.split())
            return CompletionResult(text=text, tokens=tokens, model=self._default_model(model))
        except Exception as exc:  # pragma: no cover - exercised only live
            raise ProviderError("openai", "completion failed", cause=exc)

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
