"""Tests for :mod:`furrow.config`.

The module owns three things worth pinning down:

* :data:`~furrow.config.PROVIDER_MODELS` and the ``_apply_provider_defaults``
  model validator that turn a bare ``provider=`` into a full set of model names.
  The contract is asymmetric on purpose: a field the caller passed explicitly
  wins, and only the fields they left out are filled in from the provider table.
  Expectations here are read out of :data:`PROVIDER_MODELS` rather than
  hardcoded, so the table stays the single source of truth.
* The small pydantic data models (:class:`~furrow.config.TaskModel`,
  :class:`~furrow.config.Plan`, :class:`~furrow.config.TestResult`) and their
  derived helpers (``is_empty``, ``task_map``, ``duplicate_ids``).
* :class:`~furrow.config.Settings` itself: role resolution via ``model_for``,
  the advisory ``validate_runtime`` report, and the ``ge=`` field constraints.
  It also owns the module-level ``settings`` singleton, which is created at
  *import* time from ``.env``/``FURROW_*`` — so these tests must not depend on
  its values, only prove that building other instances leaves it alone.

Hermeticity: every test runs under the ``agent_env`` fixture (applied module-wide
via the autouse ``_hermetic_env`` fixture below), which clears ``FURROW_*``,
``ANTHROPIC_API_KEY`` and ``OPENAI_API_KEY``; every direct construction passes
``_env_file=None`` so no ``.env`` file is read either. Nothing here touches the
network or needs an API key.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow import config
from furrow.config import (
    PROVIDER_MODELS,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
)

#: Every provider the defaults table has an entry for. Tests that care about
#: "all providers" iterate this instead of hardcoding a second list.
ALL_PROVIDERS: list[Provider] = list(PROVIDER_MODELS)
PROVIDER_IDS = [provider.value for provider in ALL_PROVIDERS]

#: The four model fields the provider table populates.
MODEL_FIELDS = ("model", "planner_model", "worker_model", "tester_model")


@pytest.fixture(autouse=True)
def _hermetic_env(agent_env: Path) -> Path:
    """Module-wide guard: no ``FURROW_*`` var and no provider API key is visible.

    ``agent_env`` deletes them via ``monkeypatch``; every ``Settings`` built
    below also passes ``_env_file=None``, so neither the ambient environment nor
    a stray ``.env`` file can decide a test's outcome. Returns the hermetic
    temporary workspace for tests that want to build on it.
    """
    return agent_env


def _make(make_settings: Callable[..., Settings], **overrides: object) -> Settings:
    """``make_settings`` without the ceremony, for readability in each test."""
    return make_settings(**overrides)


# --------------------------------------------------------------------------
# Provider default models
# --------------------------------------------------------------------------


def test_provider_models_table_covers_every_provider_enum_member() -> None:
    # Guards the guard: if a Provider member were missing from the table, the
    # "default models" tests below would silently stop covering it.
    assert set(PROVIDER_MODELS) == set(Provider)
    for provider, defaults in PROVIDER_MODELS.items():
        assert set(defaults) == set(MODEL_FIELDS), provider


@pytest.mark.parametrize("provider", ALL_PROVIDERS, ids=PROVIDER_IDS)
def test_each_provider_supplies_its_own_default_models(
    make_settings: Callable[..., Settings], provider: Provider
) -> None:
    settings = _make(make_settings, provider=provider)
    expected = PROVIDER_MODELS[provider]

    assert settings.provider is provider
    for field in MODEL_FIELDS:
        assert getattr(settings, field) == expected[field], field


def test_openai_and_anthropic_defaults_are_not_the_same_thing(
    make_settings: Callable[..., Settings],
) -> None:
    # The three providers really are distinct, not three copies of one table.
    per_provider = [PROVIDER_MODELS[p]["model"] for p in ALL_PROVIDERS]
    assert len(set(per_provider)) == len(per_provider)


def test_default_provider_is_anthropic(make_settings: Callable[..., Settings]) -> None:
    # ``provider`` is a plain field default, not a provider-defaults lookup, so
    # omitting it entirely must still land on anthropic's row.
    settings = _make(make_settings)
    assert settings.provider is Provider.ANTHROPIC
    assert settings.model == PROVIDER_MODELS[Provider.ANTHROPIC]["model"]


# --------------------------------------------------------------------------
# Overrides win, and only the fields left out get defaulted
# --------------------------------------------------------------------------


def test_an_explicitly_passed_model_wins() -> None:
    assert Settings(_env_file=None, provider="openai", model="mine").model == "mine"


def test_an_explicit_model_does_not_disturb_the_other_role_defaults() -> None:
    settings = Settings(_env_file=None, provider="openai", model="mine")
    expected = PROVIDER_MODELS[Provider.OPENAI]
    assert settings.planner_model == expected["planner_model"]
    assert settings.worker_model == expected["worker_model"]
    assert settings.tester_model == expected["tester_model"]


def test_a_partial_override_does_not_clobber_the_rest() -> None:
    settings = Settings(_env_file=None, provider="openai", planner_model="p")
    expected = PROVIDER_MODELS[Provider.OPENAI]
    # The one field that was passed...
    assert settings.planner_model == "p"
    # ...and every field that was not is still openai's default.
    assert settings.model == expected["model"]
    assert settings.worker_model == expected["worker_model"]
    assert settings.tester_model == expected["tester_model"]


def test_an_explicit_role_override_survives_a_different_provider() -> None:
    # Cross-provider check: an override on ollama must not leak into, or be
    # overwritten by, another provider's table.
    settings = Settings(_env_file=None, provider=Provider.OLLAMA, worker_model="local-tuned")
    assert settings.worker_model == "local-tuned"
    assert settings.model == PROVIDER_MODELS[Provider.OLLAMA]["model"]


# --------------------------------------------------------------------------
# TaskModel
# --------------------------------------------------------------------------


def test_task_defaults_to_pending() -> None:
    task = TaskModel(id="1", description="d")
    assert task.status == TaskStatus.PENDING
    assert task.status is TaskStatus.PENDING


def test_task_optional_fields_default_to_empty_or_none() -> None:
    task = TaskModel(id="1", description="d")
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None
    assert task.started_at is None
    assert task.finished_at is None
    assert task.error is None
    assert task.attempts == 0


def test_task_mutable_defaults_are_not_shared_between_instances() -> None:
    first = TaskModel(id="1", description="d")
    second = TaskModel(id="2", description="d")
    first.files.append("a.py")
    assert second.files == []


def test_task_status_round_trips_through_its_string_value() -> None:
    task = TaskModel(id="1", description="d", status="completed")
    assert task.status is TaskStatus.COMPLETED
    assert task.status == "completed"


def test_an_unknown_task_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="1", description="d", status="nearly-done")


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


def _plan(*ids: str) -> Plan:
    return Plan(
        tasks=[TaskModel(id=task_id, description=f"task {task_id}") for task_id in ids],
        rationale="r",
    )


def test_plan_with_no_tasks_is_empty() -> None:
    assert _plan().is_empty is True


def test_plan_with_tasks_is_not_empty() -> None:
    assert _plan("1").is_empty is False
    assert _plan("1", "2", "3").is_empty is False


def test_task_map_maps_id_to_task() -> None:
    plan = _plan("1", "2")
    task_map = plan.task_map()
    assert set(task_map) == {"1", "2"}
    assert task_map["1"] is plan.tasks[0]
    assert task_map["2"] is plan.tasks[1]
    assert task_map["1"].description == "task 1"


def test_task_map_is_empty_for_an_empty_plan() -> None:
    assert _plan().task_map() == {}


def test_duplicate_ids_reports_a_repeated_id() -> None:
    assert _plan("1", "1").duplicate_ids() == ["1"]


def test_duplicate_ids_is_empty_when_every_id_is_unique() -> None:
    assert _plan("1", "2", "3").duplicate_ids() == []
    assert _plan().duplicate_ids() == []


def test_duplicate_ids_lists_each_extra_occurrence() -> None:
    # Once seen, "1" is never re-added, so a third occurrence counts again.
    assert _plan("1", "1", "2", "1").duplicate_ids() == ["1", "1"]


def test_plan_requires_a_rationale() -> None:
    with pytest.raises(ValidationError):
        Plan(tasks=[])


# --------------------------------------------------------------------------
# TestResult
# --------------------------------------------------------------------------


def test_test_result_defaults() -> None:
    result = TestResult(passed=True, summary="all good")
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


def test_test_result_keeps_reported_failures() -> None:
    result = TestResult(passed=False, summary="2 failed", failures=["test_a", "test_b"])
    assert result.passed is False
    assert result.failures == ["test_a", "test_b"]


# --------------------------------------------------------------------------
# Settings.model_for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ALL_PROVIDERS, ids=PROVIDER_IDS)
@pytest.mark.parametrize("role", ["planner", "worker", "tester"])
def test_model_for_resolves_each_role_to_that_providers_model(
    make_settings: Callable[..., Settings], provider: Provider, role: str
) -> None:
    settings = _make(make_settings, provider=provider)
    assert settings.model_for(role) == PROVIDER_MODELS[provider][f"{role}_model"]
    # ...which is the same string the matching attribute holds.
    assert settings.model_for(role) == getattr(settings, f"{role}_model")


def test_model_for_returns_the_mapped_model_not_the_default_one(
    make_settings: Callable[..., Settings],
) -> None:
    settings = _make(make_settings, provider=Provider.OPENAI)
    expected = PROVIDER_MODELS[Provider.OPENAI]
    # The role model and the default ``.model`` are different entries, so a
    # role lookup is observably not a fallback to ``.model``.
    assert settings.model_for("planner") == expected["planner_model"]
    assert settings.model == expected["model"]
    assert settings.model_for("planner") != settings.model


def test_model_for_uses_an_explicit_override_when_one_was_given() -> None:
    settings = Settings(_env_file=None, provider=Provider.OPENAI, worker_model="mine")
    assert settings.model_for("worker") == "mine"


@pytest.mark.parametrize("role", ["", "planner ", "PLANNER", "reviewer", "unknown"])
def test_model_for_falls_back_to_model_for_an_unknown_role(
    make_settings: Callable[..., Settings], role: str
) -> None:
    # The role map is an exact-match lookup: only the three canonical names hit.
    settings = _make(make_settings, provider=Provider.OPENAI)
    assert settings.model_for(role) == settings.model


# --------------------------------------------------------------------------
# Settings.validate_runtime
# --------------------------------------------------------------------------


def test_validate_runtime_returns_a_list_of_strings(
    make_settings: Callable[..., Settings],
) -> None:
    warnings = _make(make_settings).validate_runtime()
    assert isinstance(warnings, list)
    assert all(isinstance(item, str) for item in warnings)


def test_validate_runtime_reports_the_missing_anthropic_key(
    make_settings: Callable[..., Settings],
) -> None:
    # Hermetic by construction: ``make_settings`` forces anthropic with no keys,
    # an existing tmp workspace, max_parallel_tasks=2 and allow_shell=False, so
    # the API key is the only thing left to complain about.
    warnings = _make(make_settings, provider=Provider.ANTHROPIC).validate_runtime()
    assert warnings == ["provider is anthropic but no API key configured"]


def test_validate_runtime_is_quiet_for_a_fully_configured_setup(
    make_settings: Callable[..., Settings],
) -> None:
    settings = _make(make_settings, provider=Provider.ANTHROPIC, anthropic_api_key="k")
    assert settings.validate_runtime() == []


def test_validate_runtime_reports_a_missing_openai_key(
    make_settings: Callable[..., Settings],
) -> None:
    settings = _make(make_settings, provider=Provider.OPENAI)
    assert "provider is openai but no API key configured" in settings.validate_runtime()
    # The anthropic key is irrelevant once the provider is openai.
    assert "anthropic" not in " ".join(settings.validate_runtime())


def test_validate_runtime_never_complains_about_keys_for_ollama(
    make_settings: Callable[..., Settings],
) -> None:
    assert _make(make_settings, provider=Provider.OLLAMA).validate_runtime() == []


def test_validate_runtime_reports_a_missing_workspace(
    make_settings: Callable[..., Settings], tmp_workspace: Path
) -> None:
    missing = tmp_workspace / "does-not-exist"
    settings = _make(make_settings, anthropic_api_key="k", workspace=missing)
    warnings = settings.validate_runtime()
    assert warnings == [f"workspace does not exist: {missing}"]


def test_validate_runtime_reports_a_large_fan_out_and_shell_access(
    make_settings: Callable[..., Settings],
) -> None:
    settings = _make(
        make_settings, anthropic_api_key="k", max_parallel_tasks=21, allow_shell=True
    )
    warnings = settings.validate_runtime()
    assert "max_parallel_tasks (21) > 20" in warnings
    assert "allow_shell is enabled" in warnings


def test_validate_runtime_is_a_pure_report_and_does_not_raise(
    make_settings: Callable[..., Settings], tmp_workspace: Path
) -> None:
    # Every warning-producing branch at once, and a call on a settings object
    # whose workspace is a Path that was never created.
    settings = _make(
        make_settings,
        provider=Provider.OPENAI,
        workspace=tmp_workspace / "gone",
        max_parallel_tasks=99,
        allow_shell=True,
    )
    warnings = settings.validate_runtime()
    assert isinstance(warnings, list)
    assert len(warnings) == 4
    # Calling it twice reports the same thing: no state is accumulated or cleared.
    assert settings.validate_runtime() == warnings


# --------------------------------------------------------------------------
# Field constraints
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_cycles", -1),
        ("max_parallel_tasks", 0),
        ("test_timeout", 0),
    ],
)
def test_out_of_range_settings_are_rejected(
    make_settings: Callable[..., Settings], field: str, value: int
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        _make(make_settings, **{field: value})
    assert field in str(excinfo.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_cycles", 0),
        ("max_parallel_tasks", 1),
        ("test_timeout", 1),
    ],
)
def test_the_inclusive_lower_bound_is_accepted(
    make_settings: Callable[..., Settings], field: str, value: int
) -> None:
    assert getattr(_make(make_settings, **{field: value}), field) == value


def test_an_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, provider="gemini")


# --------------------------------------------------------------------------
# The module-level singleton
# --------------------------------------------------------------------------


def test_the_module_level_singleton_is_a_settings_instance() -> None:
    assert isinstance(config.settings, Settings)


def test_building_other_settings_does_not_mutate_the_global_singleton(
    make_settings: Callable[..., Settings],
) -> None:
    # The singleton is built at import time from .env/FURROW_*; its *values* are
    # therefore not asserted here (they depend on the ambient environment), only
    # that constructing other instances leaves the object exactly as it was.
    global_settings = config.settings
    before_dump = global_settings.model_dump()
    before_fields_set = set(global_settings.model_fields_set)

    built = [
        _make(make_settings),
        _make(make_settings, provider=Provider.OPENAI, model="mine"),
        _make(make_settings, provider=Provider.OLLAMA, planner_model="p"),
        Settings(_env_file=None, provider="openai", model="mine"),
    ]
    # Proof the constructed instances really do differ from the singleton, so
    # the assertions below are not vacuous.
    assert all(instance is not global_settings for instance in built)
    assert any(instance.model != global_settings.model for instance in built)

    assert global_settings is config.settings
    assert global_settings.model_dump() == before_dump
    assert set(global_settings.model_fields_set) == before_fields_set


def test_the_make_settings_fixture_does_not_mutate_the_global_singleton(
    make_settings: Callable[..., Settings],
) -> None:
    before = config.settings.model_dump()
    settings = _make(make_settings, provider=Provider.OPENAI)
    assert settings.model == PROVIDER_MODELS[Provider.OPENAI]["model"]
    assert config.settings.model_dump() == before
