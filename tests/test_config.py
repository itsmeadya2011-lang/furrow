"""Configuration models: enums, coercion rules, env prefix, and field defaults."""

from __future__ import annotations

import pytest

from furrow import config


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("anthropic", config.Provider.ANTHROPIC),
        ("openai", config.Provider.OPENAI),
        ("ollama", config.Provider.OLLAMA),
    ],
)
def test_provider_accepts_the_three_documented_strings(raw: str, expected: config.Provider) -> None:
    assert config.Provider(raw) is expected


def test_provider_is_a_string_enum() -> None:
    # Code that compares a provider to a raw string (or serialises it) keeps working.
    assert config.Provider.ANTHROPIC == "anthropic"
    assert config.Provider.OLLAMA.value == "ollama"


def test_provider_rejects_an_unknown_string() -> None:
    with pytest.raises(ValueError):
        config.Provider("bedrock")


def test_task_status_members_and_values() -> None:
    assert [status.value for status in config.TaskStatus] == [
        "pending",
        "running",
        "completed",
        "failed",
        "blocked",
    ]


def test_task_status_compares_equal_to_its_string() -> None:
    assert config.TaskStatus.COMPLETED == "completed"


def test_task_model_coerces_a_plain_status_string() -> None:
    # A raw string is exactly what an LLM-produced plan contains.
    task = config.TaskModel(id="1", description="do it", status="completed")  # type: ignore[arg-type]
    assert task.status is config.TaskStatus.COMPLETED


def test_task_model_rejects_an_unknown_status() -> None:
    with pytest.raises(ValueError):
        config.TaskModel(id="1", description="do it", status="half-done")  # type: ignore[arg-type]


def test_task_model_defaults() -> None:
    task = config.TaskModel(id="1", description="do it")
    assert task.files == []
    assert task.dependencies == []
    assert task.status is config.TaskStatus.PENDING
    assert task.result is None


def test_task_model_defaults_are_not_shared() -> None:
    first = config.TaskModel(id="1", description="one")
    second = config.TaskModel(id="2", description="two")
    first.files.append("a.py")
    assert second.files == []


def test_plan_complete_defaults_to_false() -> None:
    plan = config.Plan(tasks=[config.TaskModel(id="1", description="do it")], rationale="why")
    assert plan.complete is False


def test_plan_complete_is_settable() -> None:
    plan = config.Plan(tasks=[], rationale="nothing left", complete=True)
    assert plan.complete is True


def test_test_result_failures_default_to_empty() -> None:
    # Imported through the module on purpose: a bare ``TestResult`` import into a
    # test module makes pytest try to collect the model class.
    result = config.TestResult(passed=True, summary="all good")
    assert result.failures == []


def test_settings_honours_the_furrow_env_prefix(
    monkeypatch: pytest.MonkeyPatch, settings_factory
) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "2")
    assert settings_factory().max_parallel_tasks == 2


def test_settings_ignores_unprefixed_env_vars(
    monkeypatch: pytest.MonkeyPatch, settings_factory
) -> None:
    monkeypatch.setenv("MAX_PARALLEL_TASKS", "99")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "3")
    assert settings_factory().max_parallel_tasks == 3


def test_settings_parses_a_provider_string_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, settings_factory
) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    assert settings_factory().provider is config.Provider.OLLAMA


def test_settings_parses_an_int_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, settings_factory
) -> None:
    monkeypatch.setenv("FURROW_TEST_TIMEOUT", "7")
    assert settings_factory().test_timeout == 7


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("max_retries", 2),
        ("test_timeout", 120),
        ("test_command", None),
        ("max_idle_cycles", 3),
    ],
)
def test_new_tuning_fields_have_their_documented_defaults(
    name: str, expected: object, settings_factory
) -> None:
    # Regression: these fields are what stop the loop from spinning forever.
    assert getattr(settings_factory(), name) == expected


def test_loop_control_defaults(settings_factory) -> None:
    settings = settings_factory()
    assert settings.max_cycles == 0  # 0 means "no explicit cap"
    assert settings.max_parallel_tasks == 5
    assert settings.provider is config.Provider.ANTHROPIC
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.workspace.exists()


def test_settings_constructor_overrides_beat_defaults(settings_factory) -> None:
    settings = settings_factory(test_command="pytest -q", test_timeout=5)
    assert settings.test_command == "pytest -q"
    assert settings.test_timeout == 5


def test_module_level_settings_singleton_is_a_settings_instance() -> None:
    assert isinstance(config.settings, config.Settings)


def test_module_level_settings_singleton_is_stable() -> None:
    assert config.settings is config.settings
