"""Config must fail loudly on bad input, and say "unknown" rather than guess."""

from __future__ import annotations

import pytest

from warrant.config.settings import Settings
from warrant.exceptions import ConfigError


def _load(monkeypatch, **env: str) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings.load()


def test_defaults_are_offline_and_volume_is_unknown(monkeypatch) -> None:
    for key in ("WARRANT_MOCK", "WARRANT_RUNS_PER_MONTH", "WARRANT_NOVELTY_EPSILON"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings.load()
    assert settings.mock is True                     # offline determinism by default
    assert settings.runs_per_month is None           # never a made-up traffic figure
    assert settings.novelty_epsilon == 0.15


def test_the_two_epsilons_are_separate_knobs(monkeypatch) -> None:
    """Lexical output novelty and posterior KL are different quantities."""
    settings = _load(
        monkeypatch, WARRANT_NOVELTY_EPSILON="0.3", WARRANT_NOVELTY_KL_EPSILON="0.01"
    )
    assert settings.novelty_epsilon == 0.3
    assert settings.novelty_kl_epsilon == 0.01


def test_declared_volume_is_read(monkeypatch) -> None:
    assert _load(monkeypatch, WARRANT_RUNS_PER_MONTH="50000").runs_per_month == 50_000


def test_unparseable_volume_fails_fast(monkeypatch) -> None:
    with pytest.raises(ConfigError, match="must be an int"):
        _load(monkeypatch, WARRANT_RUNS_PER_MONTH="lots")


def test_zero_volume_fails_validation(monkeypatch) -> None:
    with pytest.raises(ConfigError, match="RUNS_PER_MONTH"):
        _load(monkeypatch, WARRANT_RUNS_PER_MONTH="0").validate()


def test_unparseable_value_fails_fast(monkeypatch) -> None:
    with pytest.raises(ConfigError, match="must be a float"):
        _load(monkeypatch, WARRANT_POSTERIOR_TEMPERATURE="warm")


def test_out_of_range_values_fail_validation(monkeypatch) -> None:
    with pytest.raises(ConfigError, match="POSTERIOR_SAMPLES"):
        _load(monkeypatch, WARRANT_POSTERIOR_SAMPLES="0").validate()


def test_novelty_epsilon_must_be_a_fraction(monkeypatch) -> None:
    with pytest.raises(ConfigError, match="NOVELTY_EPSILON must be in"):
        _load(monkeypatch, WARRANT_NOVELTY_EPSILON="2.0").validate()


def test_require_live_names_every_missing_credential(monkeypatch) -> None:
    settings = _load(monkeypatch, WARRANT_MOCK="0", ANTHROPIC_API_KEY="", OPENAI_API_KEY="")
    with pytest.raises(ConfigError) as exc:
        settings.require_live()
    message = str(exc.value)
    assert "ANTHROPIC_API_KEY" in message and "OPENAI_API_KEY" in message

    # In mock mode there is nothing to require.
    _load(monkeypatch, WARRANT_MOCK="1").require_live()
