"""Provider protocol, posterior-elicitation helpers, and a deterministic mock.

Design notes
------------
* Posteriors are elicited by asking a model to emit a JSON distribution over a
  closed option set, averaged over K self-consistency samples (the paper's
  calibrated-prompt technique for models without usable logprobs). OpenAI
  logprobs are used opportunistically when available.
* ``MockProvider`` makes the entire system runnable offline and in CI. It is
  deliberately *dumb but context-sensitive*: a posterior's confidence is driven
  by lexical overlap between each option's supporting keywords (``hints``) and
  the context. This is what lets the Delegation Ledger reproduce the telephone
  game offline: when exogenous evidence is present in context the posterior is
  confident and correct; when a relay drops that evidence the posterior decays
  toward uniform.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from warrant.config import Settings, get_settings
from warrant.exceptions import ProviderError
from warrant.logging_setup import get_logger
from warrant.schemas.belief import Posterior

log = get_logger("providers")


@dataclass
class CompletionResult:
    text: str
    tokens: int
    model: str


def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Numerically stable softmax over a label->score map."""
    if temperature <= 0:
        temperature = 1e-6
    m = max(scores.values())
    exps = {k: math.exp((v - m) / temperature) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def average_posteriors(samples: list[dict[str, float]], options: list[str]) -> Posterior:
    """Average K sampled distributions into one Posterior over `options`."""
    if not samples:
        return Posterior.uniform(options)
    agg = {opt: 0.0 for opt in options}
    for s in samples:
        total = sum(max(0.0, s.get(opt, 0.0)) for opt in options)
        if total <= 0:
            for opt in options:
                agg[opt] += 1.0 / len(options)
        else:
            for opt in options:
                agg[opt] += max(0.0, s.get(opt, 0.0)) / total
    return Posterior(probs={opt: agg[opt] / len(samples) for opt in options})


def build_posterior_prompt(question: str, options: list[str], context: str) -> str:
    """Prompt asking a model to emit a JSON probability distribution."""
    opts = ", ".join(f'"{o}"' for o in options)
    return (
        "You are a calibrated classifier. Read the CONTEXT and answer the "
        "QUESTION by returning ONLY a JSON object mapping each option to a "
        "probability in [0,1] that sums to 1. Do not add prose.\n\n"
        f"OPTIONS: [{opts}]\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        'JSON (e.g. {"' + options[0] + '": 0.8, ...}):'
    )


def parse_distribution(text: str, options: list[str]) -> dict[str, float]:
    """Best-effort parse of a JSON distribution; falls back to uniform."""
    import json
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {opt: 1.0 / len(options) for opt in options}
    try:
        raw = json.loads(match.group(0))
    except Exception:
        return {opt: 1.0 / len(options) for opt in options}
    dist = {opt: float(raw.get(opt, 0.0)) for opt in options}
    if sum(dist.values()) <= 0:
        return {opt: 1.0 / len(options) for opt in options}
    return dist


@runtime_checkable
class LLMProvider(Protocol):
    """Uniform interface over Anthropic / OpenAI / mock."""

    name: str

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult: ...

    def classify_posterior(
        self,
        *,
        question: str,
        options: list[str],
        context: str,
        model: str | None = None,
        k: int | None = None,
        hints: dict[str, list[str]] | None = None,
    ) -> Posterior: ...


# --------------------------------------------------------------------------- #
# Mock provider
# --------------------------------------------------------------------------- #
class MockProvider:
    """Offline, deterministic provider for tests, CI, and the ledger probe."""

    name = "mock"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        digest = hashlib.sha1((system + "||" + prompt).encode()).hexdigest()[:8]
        text = f"[mock:{model or 'default'}:{digest}] {prompt[:280]}"
        return CompletionResult(text=text, tokens=len(prompt.split()) + 16, model=model or "mock")

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
        if not options:
            raise ProviderError("mock", "classify_posterior requires >=1 option")
        ctx = context.lower()
        hints = hints or {}
        # Base score: weak lexical presence of the option label itself.
        scores: dict[str, float] = {}
        for opt in options:
            score = 0.4 if opt.lower() in ctx else 0.0
            for kw in hints.get(opt, []):
                kw_l = kw.lower().strip()
                if kw_l and kw_l in ctx:
                    # Strong evidence: exogenous keyword present in context.
                    score += 3.0
            scores[opt] = score
        # Deterministic tie-break jitter so uniform cases aren't perfectly flat.
        seed = int(hashlib.sha1((question + context[:64]).encode()).hexdigest(), 16)
        for i, opt in enumerate(options):
            scores[opt] += ((seed >> (i * 5)) % 7) * 0.01
        return Posterior(probs=softmax(scores, temperature=0.5))


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #
def get_provider(role: str, settings: Settings | None = None) -> LLMProvider:
    """Return the provider for a role: 'orchestrator' or 'worker'.

    In mock mode both roles resolve to :class:`MockProvider`.
    """
    settings = settings or get_settings()
    if settings.mock:
        return MockProvider(settings)
    if role == "orchestrator":
        from warrant.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    if role == "worker":
        from warrant.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    raise ProviderError("factory", f"unknown provider role {role!r}")
