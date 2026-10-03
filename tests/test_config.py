"""Unit tests for ``furrow.config``.

Protects the settings contract the rest of the package relies on: the
``FURROW_``-prefixed environment interface, the documented defaults for the
fields the orchestrator, tester and LLM client read, ``model_copy`` as the
supported way to derive an independent copy (what the CLI does so the global
``furrow.config.settings`` singleton is never mutated), and the ``Provider``
enum used for dispatch.

Nothing here mutates the global singleton or touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import Provider, Settings
from furrow.config import settings as global_settings


def _defaults() -> Settings:
    return Settings(_env_file=None)


def test_env_prefix_is_furrow() -> None:
    assert Settings.model_config["env_prefix"] == "FURROW_"


def test_env_var_with_the_prefix_overrides_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_TOKENS", "1234")
    assert Settings(_env_file=None).max_tokens == 1234


def test_env_var_without_the_prefix_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_TOKENS", "9999")
    assert Settings(_env_file=None).max_tokens == 4096


def test_env_var_can_select_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    assert Settings(_env_file=None).provider is Provider.OLLAMA


def test_workspace_default_is_the_current_directory() -> None:
    assert _defaults().workspace == Path.cwd()


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("max_tokens", 4096),
        ("request_timeout", 120.0),
        ("llm_max_retries", 3),
        ("test_command", None),
        ("test_timeout", 120.0),
        ("max_parallel_tasks", 5),
        ("max_cycles", 0),
    ],
)
def test_documented_defaults(field: str, expected: object) -> None:
    assert getattr(_defaults(), field) == expected


def test_api_keys_default_to_unset_and_ollama_points_at_localhost() -> None:
    settings = _defaults()
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"


def test_max_cycles_zero_means_unlimited() -> None:
    # 0 is the sentinel for "keep cycling"; the orchestrator must not treat it as zero.
    assert _defaults().max_cycles <= 0


def test_model_copy_update_produces_an_independent_copy() -> None:
    updated = global_settings.model_copy(update={"max_tokens": 7, "provider": Provider.OLLAMA})
    assert updated is not global_settings
    assert updated.max_tokens == 7
    assert updated.provider is Provider.OLLAMA
    # The global singleton is untouched, so other consumers keep the defaults.
    assert global_settings.max_tokens == 4096
    assert global_settings.provider is not Provider.OLLAMA
    updated.llm_max_retries = 99
    assert global_settings.llm_max_retries == 3, "copies must not share mutable state"


def test_settings_reject_a_value_of_the_wrong_type() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_tokens="not-an-int")


def test_provider_values() -> None:
    assert {member.value for member in Provider} == {"anthropic", "openai", "ollama"}
    assert [member.name for member in Provider] == ["ANTHROPIC", "OPENAI", "OLLAMA"]


def test_provider_lookups_round_trip() -> None:
    for member in Provider:
        assert Provider(member.value) is member
    assert Provider.ANTHROPIC == "anthropic"
    with pytest.raises(ValueError):
        Provider("bedrock")