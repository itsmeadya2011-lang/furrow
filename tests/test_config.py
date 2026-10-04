"""Contract tests for :mod:`furrow.config`: provider selection and settings resolution.

Three behaviours here are easy to break and expensive to debug when they do:

* **Environment aliasing.** Because ``env_prefix="FURROW_"`` is global, the key fields
  declare explicit ``AliasChoices`` so both ``FURROW_ANTHROPIC_API_KEY`` and the
  README-documented bare ``ANTHROPIC_API_KEY`` work, prefixed first.
* **Call-time CWD resolution.** ``workspace`` is ``None`` by default and
  :meth:`Settings.workspace_path` resolves it when called, so a process that changes
  directory after import still sees the right root.
* **The cache.** ``get_settings`` is ``lru_cache``d. Tests that change the environment must
  pass ``refresh=True``, and ``get_settings()`` without a rebuild must keep returning the
  previously built instance -- asserted explicitly so nobody "optimises" the cache away.

Defaults are asserted against ``plain()`` so a developer's local ``.env``
cannot turn a default test into a coin flip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from furrow.config import Provider, Settings, get_settings


def plain(**overrides: Any) -> Settings:
    """Build ``Settings`` with dotenv loading switched off.

    ``Settings(env_file=".env")`` is resolved against the process CWD, so a plain
    ``Settings()`` in a test would silently pick up a developer's local ``.env`` and turn
    every "default" assertion into a coin flip. The autouse ``hermetic_env`` fixture already
    disables dotenv for the whole session; passing ``_env_file=None`` as well keeps each
    call site honest about what it is asserting.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]

# ------------------------------------------------------------------------------------- Provider


def test_provider_values():
    assert [provider.value for provider in Provider] == ["anthropic", "openai", "ollama"]


def test_provider_is_a_str_mixin():
    assert isinstance(Provider.ANTHROPIC, str)
    assert Provider.OPENAI == "openai"
    assert Provider("ollama") is Provider.OLLAMA


@pytest.mark.parametrize(
    ("provider", "env_var"),
    [
        (Provider.ANTHROPIC, "ANTHROPIC_API_KEY"),
        (Provider.OPENAI, "OPENAI_API_KEY"),
        (Provider.OLLAMA, None),
    ],
)
def test_provider_env_var(provider: Provider, env_var: str | None):
    assert provider.env_var == env_var


# ------------------------------------------------------------------------------------- defaults


def test_settings_defaults():
    settings = plain()

    assert settings.provider is Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.planner_model == "claude-sonnet-4-20250514"
    assert settings.worker_model == "claude-sonnet-4-20250514"
    assert settings.tester_model == "claude-sonnet-4-20250514"
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "llama3.1"
    assert settings.workspace is None
    assert settings.max_parallel_tasks == 4
    assert settings.max_tokens == 8192
    assert settings.test_command is None
    assert settings.test_timeout == 120
    assert settings.log_level == "INFO"
    assert settings.log_json is False


def test_max_cycles_defaults_to_zero_which_means_unlimited():
    # 0 is the "keep going" sentinel, not "do nothing"; the orchestrator loops while
    # `settings.max_cycles` is falsy.
    assert plain().max_cycles == 0


# ------------------------------------------------------------------------------- workspace_path


def test_workspace_path_resolves_a_relative_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = plain(workspace="sub/project")

    resolved = settings.workspace_path()

    assert resolved.is_absolute()
    assert resolved == (tmp_path / "sub" / "project").resolve()
    # relative in, absolute out -- agents join repo-relative paths onto this
    assert settings.workspace is not None
    assert not settings.workspace.is_absolute()


def test_workspace_path_defaults_to_the_cwd_at_call_time(tmp_path: Path, monkeypatch):
    settings = plain()
    assert settings.workspace is None

    monkeypatch.chdir(tmp_path)
    assert settings.workspace_path() == tmp_path.resolve()

    # Resolution happens per call, so a later chdir is picked up by the same instance --
    # this is what keeps a process-wide Settings usable after the CLI changes directory.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert settings.workspace_path() == elsewhere.resolve()


def test_workspace_path_expands_a_user_relative_workspace(monkeypatch):
    monkeypatch.setenv("HOME", "/home/furrow-user")
    settings = plain(workspace="~/project")

    assert settings.workspace_path() == Path("/home/furrow-user/project")


# ------------------------------------------------------------------------------------- api keys


def test_ensure_api_key_raises_naming_the_env_vars():
    settings = plain(provider=Provider.ANTHROPIC)

    with pytest.raises(ValueError) as excinfo:
        settings.ensure_api_key()

    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "FURROW_ANTHROPIC_API_KEY" in message


def test_ensure_api_key_raises_for_openai_too():
    settings = plain(provider=Provider.OPENAI)

    with pytest.raises(ValueError) as excinfo:
        settings.ensure_api_key()

    assert "FURROW_OPENAI_API_KEY" in str(excinfo.value)


def test_ensure_api_key_returns_an_empty_string_for_ollama():
    # Ollama is a local daemon: no credential, and the result still has to be safe to hand
    # to a client constructor.
    settings = plain(provider=Provider.OLLAMA)

    assert settings.ensure_api_key() == ""
    assert settings.api_key() is None
    assert "does not require an API key" in settings.missing_api_key_error()


def test_ensure_api_key_returns_the_configured_key():
    settings = plain(anthropic_api_key="sk-ant-configured")

    assert settings.ensure_api_key() == "sk-ant-configured"


def test_api_key_returns_the_active_providers_credential():
    settings = plain(
        anthropic_api_key="sk-ant-value",
        openai_api_key="sk-openai-value",
    )
    assert settings.provider is Provider.ANTHROPIC
    assert settings.api_key() == "sk-ant-value"

    as_openai = settings.model_copy(update={"provider": Provider.OPENAI})
    assert as_openai.api_key() == "sk-openai-value"

    as_ollama = settings.model_copy(update={"provider": Provider.OLLAMA})
    assert as_ollama.api_key() is None


def test_the_api_key_never_appears_in_a_dump():
    # Structured logs serialise settings; `exclude=True` on the key fields is what keeps a
    # credential out of every log line and every CLI render.
    settings = plain(
        anthropic_api_key="sk-ant-super-secret",
        openai_api_key="sk-openai-super-secret",
    )

    dumped = settings.model_dump()
    assert "anthropic_api_key" not in dumped
    assert "openai_api_key" not in dumped
    assert "super-secret" not in str(dumped)
    assert "super-secret" not in settings.model_dump_json()
    # the value is still on the model; only serialisation hides it
    assert settings.anthropic_api_key == "sk-ant-super-secret"


def test_env_aliasing_for_the_bare_anthropic_variable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bare-key")

    assert get_settings(refresh=True).anthropic_api_key == "bare-key"


def test_env_aliasing_for_the_prefixed_anthropic_variable(monkeypatch):
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "prefixed-key")

    assert get_settings(refresh=True).anthropic_api_key == "prefixed-key"


def test_the_prefixed_anthropic_variable_wins_when_both_are_set(monkeypatch):
    # AliasChoices order is the precedence order: the explicit FURROW_ form beats the bare
    # one, so a per-run override is not shadowed by a shell export.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bare-key")
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "prefixed-key")

    assert get_settings(refresh=True).anthropic_api_key == "prefixed-key"


def test_env_aliasing_for_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "bare-openai")
    assert get_settings(refresh=True).openai_api_key == "bare-openai"

    monkeypatch.setenv("FURROW_OPENAI_API_KEY", "prefixed-openai")
    assert get_settings(refresh=True).openai_api_key == "prefixed-openai"


def test_a_prefixed_setting_parses_into_its_enum(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")

    settings = get_settings(refresh=True)

    assert settings.provider is Provider.OPENAI
    assert settings.max_parallel_tasks == 9


# ------------------------------------------------------------------------- model selection helpers


def test_models_returns_the_three_role_keys():
    settings = plain(planner_model="p-model", worker_model="w-model")

    assert settings.models() == {
        "planner": "p-model",
        "worker": "w-model",
        "tester": settings.tester_model,
    }
    assert set(settings.models()) == {"planner", "worker", "tester"}


def test_resolved_model_prefers_the_override():
    settings = plain(model="fallback-model")

    assert settings.resolved_model() == "fallback-model"
    assert settings.resolved_model("explicit-model") == "explicit-model"
    # `override or self.model`, so an empty override is not a request for an empty model.
    assert settings.resolved_model("") == "fallback-model"


# --------------------------------------------------------------------------------- cached accessor


def test_get_settings_is_cached_until_refresh_is_requested(monkeypatch):
    first = get_settings(refresh=True)
    assert get_settings() is first

    monkeypatch.setenv("FURROW_MAX_CYCLES", "99")

    # Without a refresh the env change is invisible: this is the whole reason every env
    # mutating test in this suite passes refresh=True.
    assert get_settings() is first
    assert first.max_cycles == 0

    rebuilt = get_settings(refresh=True)
    assert rebuilt is not first
    assert rebuilt.max_cycles == 99


def test_get_settings_rebuilds_when_the_provider_changes(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")

    assert get_settings(refresh=True).provider is Provider.OLLAMA
