"""Tests for configuration loading.

The settings object is process-global in practice, so these tests pin the
defaults (to catch accidental drift), the environment variable spellings (the
bare vendor names used to be ignored) and the masking of API keys in `repr`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    WorkerResult,
    get_settings,
    reset_settings_cache,
)


def test_defaults() -> None:
    # `_env_file=None` so a developer's own `.env` cannot change these numbers.
    settings = Settings(_env_file=None)
    assert settings.provider is Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-4-5"
    assert settings.planner_model is None
    assert settings.worker_model is None
    assert settings.tester_model is None
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "llama3.1"
    assert settings.max_parallel_tasks == 5
    assert settings.max_parallel_waves == 3
    assert settings.max_cycles == 0
    assert settings.workspace == Path.cwd()
    assert settings.log_level == "INFO"
    assert settings.request_timeout == 120.0
    assert settings.test_command_timeout == 300.0
    assert settings.max_tokens == 8192
    assert settings.llm_max_retries == 3


def test_model_for_falls_back_to_global_model() -> None:
    settings = Settings(_env_file=None, model="only-model")
    assert settings.model_for("planner") == "only-model"
    assert settings.model_for("worker") == "only-model"
    assert settings.model_for("tester") == "only-model"


def test_model_for_prefers_role_override() -> None:
    settings = Settings(
        _env_file=None,
        model="only-model",
        planner_model="p",
        worker_model="w",
        tester_model="t",
    )
    assert settings.model_for("planner") == "p"
    assert settings.model_for("worker") == "w"
    assert settings.model_for("tester") == "t"


def test_model_for_empty_role_override_falls_back() -> None:
    # `explicit or self.model` means an empty string behaves like None and falls
    # back to the global model. That is the desired behaviour here: an empty
    # override cannot be a valid model id, and raising would break `--model ""`
    # handling more confusingly than silently using the global model.
    settings = Settings(_env_file=None, model="only-model", worker_model="")
    assert settings.model_for("worker") == "only-model"


def test_bare_anthropic_api_key_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    # The README tells users to export bare ANTHROPIC_API_KEY, which the old
    # env_prefix="FURROW_" silently ignored.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-test"


def test_furrow_prefixed_anthropic_api_key_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "sk-prefixed")
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-prefixed"


def test_prefixed_api_key_wins_when_both_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-bare")
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "sk-prefixed")
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key.get_secret_value() == "sk-prefixed"


def test_openai_api_key_binds_from_both_spellings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-bare")
    assert Settings(_env_file=None).openai_api_key.get_secret_value() == "sk-openai-bare"

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("FURROW_OPENAI_API_KEY", "sk-openai-prefixed")
    assert Settings(_env_file=None).openai_api_key.get_secret_value() == "sk-openai-prefixed"


def test_repr_does_not_leak_the_api_key() -> None:
    # Regression test: the key used to sit in the settings repr, so it ended up
    # in tracebacks and structured logs whenever the object was dumped.
    settings = Settings(_env_file=None, anthropic_api_key=SecretStr("sk-supersecret"))
    dumped = repr(settings)
    assert "sk-supersecret" not in dumped
    assert "anthropic_api_key" not in dumped

    settings.openai_api_key = SecretStr("sk-openai-secret")
    assert "sk-openai-secret" not in repr(settings)

    # It survives validation (which would reject None being assigned to it).
    settings.max_parallel_tasks = 9
    assert settings.max_parallel_tasks == 9


def test_prefixed_env_binds_for_ordinary_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    assert Settings(_env_file=None).max_parallel_tasks == 9


def test_env_file_is_read_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("FURROW_MAX_PARALLEL_WAVES=7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert Settings().max_parallel_waves == 7


def test_max_cycles_bounds() -> None:
    # 0 is the "unlimited" sentinel and must stay legal; negative values are
    # meaningless (neither "unlimited" nor "no cycles") so they are rejected.
    assert Settings(_env_file=None, max_cycles=0).max_cycles == 0
    assert Settings(_env_file=None, max_cycles=5).max_cycles == 5
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_cycles=-1)


def test_other_numeric_bounds() -> None:
    for field in ("max_parallel_tasks", "max_parallel_waves"):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **{field: 0})
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_max_retries=-1)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, request_timeout=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, test_command_timeout=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_tokens=0)


def test_provider_is_case_insensitive() -> None:
    assert Settings(_env_file=None, provider="Anthropic").provider is Provider.ANTHROPIC
    assert Settings(_env_file=None, provider="OPENAI").provider is Provider.OPENAI
    assert Settings(_env_file=None, provider=" ollama ").provider is Provider.OLLAMA


def test_unknown_provider_is_rejected_with_a_useful_message() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, provider="not-a-provider")


def test_provider_binds_from_env_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "OpenAI")
    assert Settings(_env_file=None).provider is Provider.OPENAI


def test_get_settings_is_cached_and_resettable(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    assert get_settings() is first

    reset_settings_cache()
    second = get_settings()
    assert second is not first

    # The accessor is the only place settings get built, so it must observe the
    # environment at call time rather than at import time.
    monkeypatch.setenv("FURROW_MODEL", "from-env")
    reset_settings_cache()
    assert get_settings().model == "from-env"


def test_unknown_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_NOT_A_SETTING", "whatever")
    assert Settings(_env_file=None).model == "claude-sonnet-4-5"


def test_domain_models_are_re_exported() -> None:
    # Other modules import these from furrow.config; the definitions now live in
    # furrow.models but the old import path must keep working.
    from furrow import models

    assert Plan is models.Plan
    assert TaskModel is models.TaskModel
    assert TestResult is models.TestResult
    assert WorkerResult is models.WorkerResult
    assert TaskStatus is models.TaskStatus
