"""Behavioural tests for :mod:`furrow.config`.

Every test here is hermetic:

* ``FURROW_*`` and the two bare provider-native API-key variables are removed for the
  duration of each test by the autouse ``_hermetic_env`` fixture, so an ambient value
  can never satisfy an assertion about a default.  ``monkeypatch`` restores the
  previous environment on teardown, so no test depends on another one's side effects.
* A ``.env`` file in the process working directory can never leak in by accident: plain
  construction goes through the shared ``make_settings`` factory (``_env_file=None``),
  and the dotenv tests either pass an explicit ``_env_file`` under ``tmp_path`` or
  ``monkeypatch.chdir(tmp_path)`` before building ``Settings``.
* No network access, no module-level mutation and no reliance on the real CWD: each
  test builds the objects it needs.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from furrow.config import (
    ANTHROPIC_DEFAULT_MODELS,
    MODEL_ROLE_FIELDS,
    PRISTINE_ANTHROPIC_MODELS,
    PROVIDER_DEFAULT_MODELS,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    settings,
)

#: Variables that would be read by ``Settings`` and therefore must not leak in.  The
#: ``FURROW_*`` family comes from ``env_prefix``; the two bare spellings are stripped
#: because the API-key fields list them as aliases, so an ambient key would otherwise
#: defeat the "defaults to ``None``" assertions below.
BARE_API_KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

#: Declared ``Settings`` defaults for the bounded numeric fields.
DECLARED_BOUNDS_DEFAULTS = (
    ("request_timeout", 120.0),
    ("max_tokens", 4096),
    ("max_parallel_tasks", 5),
    ("max_cycles", 0),
)

#: Values that must be rejected both at construction and on assignment.
BOUND_VIOLATIONS: list[tuple[str, object]] = [
    ("request_timeout", 0),
    ("request_timeout", -1.5),
    ("max_tokens", 0),
    ("max_parallel_tasks", 0),
    ("max_cycles", -1),
]

#: Boundary values that must be accepted, with the value the field should end up with.
BOUND_ACCEPTANCES: list[tuple[str, object, object]] = [
    ("request_timeout", 0.001, 0.001),
    ("max_tokens", 1, 1),
    ("max_parallel_tasks", 1, 1),
    ("max_cycles", 0, 0),
]

#: The exact model id each provider is expected to resolve to, per role.  Written out
#: as literals rather than read from ``PROVIDER_DEFAULT_MODELS`` so the table itself is
#: under test and cannot pass by definition.  Unannotated on purpose: the entries are
#: ``pytest.param`` objects, which is what gives them their ``id=`` labels.
PROVIDER_DEFAULT_CASES = [
    pytest.param(
        Provider.ANTHROPIC,
        {
            "model": "claude-sonnet-4-20250514",
            "planner_model": "claude-3-5-haiku-20241022",
            "worker_model": "claude-3-5-sonnet-20241022",
            "tester_model": "claude-3-5-sonnet-20241022",
        },
        id="anthropic",
    ),
    pytest.param(
        Provider.OPENAI,
        {
            "model": "gpt-4o",
            "planner_model": "gpt-4o-mini",
            "worker_model": "gpt-4o",
            "tester_model": "gpt-4o",
        },
        id="openai",
    ),
    pytest.param(
        Provider.OLLAMA,
        {
            "model": "qwen2.5-coder",
            "planner_model": "qwen2.5",
            "worker_model": "qwen2.5-coder",
            "tester_model": "qwen2.5",
        },
        id="ollama",
    ),
]


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every environment variable ``Settings`` can read, for one test.

    ``monkeypatch`` restores the original values on teardown.  The ``.env`` half of the
    isolation is handled per test (``_env_file=None``, an explicit ``_env_file`` path,
    or ``monkeypatch.chdir``), never by changing the process CWD globally.
    """
    for name in [env for env in os.environ if env.startswith("FURROW_")]:
        monkeypatch.delenv(name)
    for name in BARE_API_KEY_VARS:
        monkeypatch.delenv(name, raising=False)


@contextmanager
def _warnings_are_errors() -> Iterator[None]:
    """Turn any warning raised inside the block into an exception, then restore filters."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _write_dotenv(directory: Path, body: str) -> Path:
    """Write ``body`` to ``directory/.env`` and return the path.

    Always UTF-8 to match the declared ``env_file_encoding`` of :class:`Settings`.
    """
    env_file = directory / ".env"
    env_file.write_text(body, encoding="utf-8")
    return env_file


def _completed_task() -> TaskModel:
    """A task in the ``COMPLETED`` state, the shape every dump test needs."""
    return TaskModel(id="1", description="write the thing", status=TaskStatus.COMPLETED)


def _settings_from_env_file(env_file: Path | None) -> Settings:
    """Build :class:`Settings` reading dotenv values from *env_file* (``None`` = skip).

    ``_env_file`` is a ``BaseSettings`` constructor keyword that works at runtime but
    that mypy cannot see on the generated ``__init__`` signature, so the single ignore
    below stands in for every dotenv call site in this module.
    """
    return Settings(_env_file=env_file)  # type: ignore[call-arg]


# --------------------------------------------------------------------------------------
# TaskStatus
# --------------------------------------------------------------------------------------


def test_task_status_subclasses_str() -> None:
    assert issubclass(TaskStatus, str)
    assert isinstance(TaskStatus.PENDING, str)


def test_task_status_compares_equal_to_its_string_value() -> None:
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"


def test_task_status_is_not_equal_to_a_different_status_value() -> None:
    assert TaskStatus.PENDING != "completed"


def test_task_status_membership_in_a_tuple_of_raw_strings_still_works() -> None:
    assert TaskStatus.COMPLETED in ("pending", "completed")


def test_task_status_renders_as_its_bare_value_not_its_repr() -> None:
    # The explicit __str__ is what stops logs and prompts from reading "TaskStatus.PENDING".
    assert f"{TaskStatus.PENDING}" == "pending"
    assert f"{TaskStatus.COMPLETED}" == "completed"
    # The %-form is a distinct str-machinery path from f-string/format; kept on purpose.
    assert "%s" % TaskStatus.FAILED == "failed"  # noqa: UP031
    assert str(TaskStatus.FAILED) == "failed"


def test_task_status_has_exactly_the_three_lifecycle_members() -> None:
    assert {member.name for member in TaskStatus} == {"PENDING", "COMPLETED", "FAILED"}
    assert {member.value for member in TaskStatus} == {"pending", "completed", "failed"}


@pytest.mark.parametrize("member", list(TaskStatus), ids=[m.value for m in TaskStatus])
def test_task_status_round_trips_through_its_value(member: TaskStatus) -> None:
    assert TaskStatus(member.value) is member


def test_task_status_rejects_an_unknown_value() -> None:
    with pytest.raises(ValueError):
        TaskStatus("done")


# --------------------------------------------------------------------------------------
# TaskModel
# --------------------------------------------------------------------------------------


def test_task_model_defaults_to_pending_with_no_result() -> None:
    task = TaskModel(id="1", description="d")
    assert task.status is TaskStatus.PENDING
    assert task.result is None


def test_task_model_files_and_dependencies_default_to_empty_lists() -> None:
    task = TaskModel(id="1", description="d")
    assert task.files == []
    assert task.dependencies == []


def test_task_model_files_are_not_shared_between_instances() -> None:
    # A shared mutable default would let one task's files leak into every other task.
    first = TaskModel(id="1", description="d")
    second = TaskModel(id="2", description="d")
    first.files.append("a.py")
    assert second.files == []
    assert first.files is not second.files


def test_task_model_dependencies_are_not_shared_between_instances() -> None:
    first = TaskModel(id="1", description="d")
    second = TaskModel(id="2", description="d")
    first.dependencies.append("1")
    assert second.dependencies == []


def test_task_model_coerces_a_status_string_on_construction() -> None:
    # The raw string is the point: TaskModel.validate_assignment must coerce it.
    task = TaskModel(id="1", description="d", status="completed")  # type: ignore[arg-type]
    assert task.status is TaskStatus.COMPLETED


def test_task_model_rejects_an_unknown_status_on_construction() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="1", description="d", status="bogus")  # type: ignore[arg-type]


def test_task_model_assignment_coerces_a_status_string() -> None:
    task = TaskModel(id="1", description="d")
    task.status = "completed"  # type: ignore[assignment]
    assert task.status is TaskStatus.COMPLETED


def test_task_model_assignment_rejects_an_unknown_status_and_keeps_the_old_value() -> None:
    task = TaskModel(id="1", description="d", status=TaskStatus.COMPLETED)
    with pytest.raises(ValidationError):
        task.status = "bogus"  # type: ignore[assignment]
    assert task.status is TaskStatus.COMPLETED


def test_task_model_requires_an_id_and_a_description() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="1")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        TaskModel(description="d")  # type: ignore[call-arg]


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------


def test_model_dump_emits_no_user_warning_and_keeps_the_enum_member() -> None:
    task = _completed_task()
    with _warnings_are_errors():
        payload = task.model_dump()
    assert payload["status"] is TaskStatus.COMPLETED


def test_model_dump_in_json_mode_emits_no_user_warning_and_yields_a_plain_string() -> None:
    task = _completed_task()
    with _warnings_are_errors():
        payload = task.model_dump(mode="json")
    assert payload["status"] == "completed"
    assert type(payload["status"]) is str


def test_model_dump_json_emits_no_user_warning_and_serialises_the_status_as_a_string() -> None:
    task = _completed_task()
    with _warnings_are_errors():
        raw = task.model_dump_json()
    assert '"status":"completed"' in raw


def test_model_dump_json_round_trips_back_into_a_task_model() -> None:
    task = _completed_task()
    with _warnings_are_errors():
        raw = task.model_dump_json()
    assert TaskModel.model_validate_json(raw).status is TaskStatus.COMPLETED


# --------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------


def test_plan_holds_its_tasks() -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert plan.rationale == "ok"
    assert plan.tasks[0].description == "do thing"


def test_plan_accepts_an_empty_task_list() -> None:
    assert Plan(tasks=[], rationale="nothing to do").tasks == []


@pytest.mark.parametrize(
    ("kwargs", "missing"),
    [
        pytest.param({"tasks": []}, "rationale", id="missing-rationale"),
        pytest.param({"rationale": "r"}, "tasks", id="missing-tasks"),
    ],
)
def test_plan_requires_both_tasks_and_rationale(kwargs: dict[str, Any], missing: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Plan(**kwargs)
    assert missing in str(excinfo.value)


def test_plan_coerces_task_dictionaries_into_task_models() -> None:
    # The raw dict is deliberate: dict -> TaskModel coercion is the behaviour under test.
    plan = Plan(tasks=[{"id": "9", "description": "d"}], rationale="r")  # type: ignore[list-item]
    assert isinstance(plan.tasks[0], TaskModel)
    assert plan.tasks[0].status is TaskStatus.PENDING


# --------------------------------------------------------------------------------------
# TestResult
# --------------------------------------------------------------------------------------


def test_test_result_passing_result_needs_no_failures() -> None:
    result = TestResult(passed=True, summary="ok")
    assert result.failures == []


def test_test_result_failures_are_not_shared_between_instances() -> None:
    first = TestResult(passed=False, summary="boom")
    second = TestResult(passed=False, summary="boom")
    first.failures.append("tests/test_a.py::test_x")
    assert second.failures == []


def test_test_result_records_reported_failures() -> None:
    result = TestResult(passed=False, summary="1 failed", failures=["tests/test_a.py::test_x"])
    assert result.failures == ["tests/test_a.py::test_x"]


def test_test_result_is_flagged_as_not_a_pytest_test_class() -> None:
    # Without __test__ = False, pytest collects "TestResult" and errors on __init__.
    assert TestResult.__test__ is False


@pytest.mark.parametrize(
    ("kwargs", "missing"),
    [
        pytest.param({"summary": "s"}, "passed", id="missing-passed"),
        pytest.param({"passed": True}, "summary", id="missing-summary"),
    ],
)
def test_test_result_requires_passed_and_summary(kwargs: dict[str, Any], missing: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        TestResult(**kwargs)
    assert missing in str(excinfo.value)


# --------------------------------------------------------------------------------------
# Settings.model_config
# --------------------------------------------------------------------------------------


def test_settings_reads_the_furrow_prefixed_environment() -> None:
    assert Settings.model_config["env_prefix"] == "FURROW_"


def test_settings_reads_an_env_file_by_default() -> None:
    assert Settings.model_config["env_file"] == ".env"


def test_settings_encodes_the_env_file_as_utf8() -> None:
    assert Settings.model_config["env_file_encoding"] == "utf-8"


def test_settings_validates_on_assignment() -> None:
    assert Settings.model_config["validate_assignment"] is True


def test_settings_ignores_extra_keys() -> None:
    assert Settings.model_config["extra"] == "ignore"


def test_settings_can_be_populated_by_field_name() -> None:
    assert Settings.model_config["populate_by_name"] is True


def test_unknown_keyword_arguments_are_ignored(
    make_settings: Callable[..., Settings]
) -> None:
    config = make_settings(not_a_real_setting=123)
    assert not hasattr(config, "not_a_real_setting")


def test_api_keys_can_be_passed_by_field_name_despite_the_validation_alias(
    make_settings: Callable[..., Settings]
) -> None:
    # populate_by_name is what keeps the field names usable as init kwargs even though
    # the fields declare a validation_alias.
    config = make_settings(anthropic_api_key="by-name-a", openai_api_key="by-name-o")
    assert config.anthropic_api_key == "by-name-a"
    assert config.openai_api_key == "by-name-o"


# --------------------------------------------------------------------------------------
# Provider default resolution
# --------------------------------------------------------------------------------------


def test_provider_compares_equal_to_its_string_value() -> None:
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider("ollama") is Provider.OLLAMA


def test_provider_has_exactly_the_three_supported_members() -> None:
    assert {member.value for member in Provider} == {"anthropic", "openai", "ollama"}


def test_provider_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError):
        Provider("bedrock")


def test_anthropic_is_the_default_provider(make_settings: Callable[..., Settings]) -> None:
    assert make_settings().provider is Provider.ANTHROPIC


def test_provider_can_be_supplied_as_a_bare_string(
    make_settings: Callable[..., Settings]
) -> None:
    assert make_settings(provider="ollama").provider is Provider.OLLAMA


@pytest.mark.parametrize(("provider", "expected"), PROVIDER_DEFAULT_CASES)
def test_provider_defaults_set_every_model_role(
    provider: Provider,
    expected: dict[str, str],
    make_settings: Callable[..., Settings],
) -> None:
    config = make_settings(provider=provider)
    for role, model in expected.items():
        assert getattr(config, role) == model, f"{provider.value}.{role}"


@pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.OLLAMA], ids=["openai", "ollama"])
def test_no_claude_model_survives_a_non_anthropic_provider(
    provider: Provider, make_settings: Callable[..., Settings]
) -> None:
    # The bug this guards against: FURROW_PROVIDER=openai shipping claude-sonnet-4-...
    # to the OpenAI API, which fails upstream as an opaque model_not_found.
    config = make_settings(provider=provider)
    for role in MODEL_ROLE_FIELDS:
        assert not getattr(config, role).startswith("claude-")


def test_an_explicitly_configured_model_survives_a_provider_switch(
    make_settings: Callable[..., Settings]
) -> None:
    config = make_settings(provider=Provider.OPENAI, model="my-model")
    assert config.model == "my-model"


def test_an_explicit_model_leaves_untouched_siblings_remapped(
    make_settings: Callable[..., Settings]
) -> None:
    config = make_settings(provider=Provider.OPENAI, model="my-model")
    assert config.planner_model == "gpt-4o-mini"
    assert config.worker_model == "gpt-4o"
    assert config.tester_model == "gpt-4o"


def test_assigning_provider_after_construction_remaps_the_models(
    make_settings: Callable[..., Settings]
) -> None:
    config = make_settings()
    assert config.model == "claude-sonnet-4-20250514"
    config.provider = Provider.OPENAI
    assert config.model == "gpt-4o"
    assert config.planner_model == "gpt-4o-mini"


def test_assigning_a_different_model_role_re_runs_the_provider_defaults(
    make_settings: Callable[..., Settings]
) -> None:
    # validate_assignment re-runs the model_validator(mode="after"), so re-introducing a
    # pristine Anthropic id into a role is remapped straight back.
    config = make_settings(provider=Provider.OPENAI)
    config.model = "claude-sonnet-4-20250514"
    assert config.model == "gpt-4o"


def test_assigning_provider_back_to_anthropic_does_not_restore_claude_defaults(
    make_settings: Callable[..., Settings]
) -> None:
    # The remapping is one-way: going back to the default provider leaves whatever the
    # previous provider resolved to in place rather than re-seeding the declared ids.
    config = make_settings(provider=Provider.OPENAI)
    config.provider = Provider.ANTHROPIC
    assert config.model == "gpt-4o"
    assert config.planner_model == "gpt-4o-mini"


def test_assigning_an_unrelated_field_leaves_the_models_alone(
    make_settings: Callable[..., Settings]
) -> None:
    config = make_settings(provider=Provider.OPENAI, model="my-model")
    config.max_tokens = 111
    assert config.model == "my-model"
    assert config.max_tokens == 111


def test_provider_default_tables_cover_every_provider_and_every_role() -> None:
    assert set(PROVIDER_DEFAULT_MODELS) == set(Provider)
    for defaults in PROVIDER_DEFAULT_MODELS.values():
        assert set(defaults) == set(MODEL_ROLE_FIELDS)


def test_anthropic_default_table_is_the_declared_pristine_table() -> None:
    assert PROVIDER_DEFAULT_MODELS[Provider.ANTHROPIC] == ANTHROPIC_DEFAULT_MODELS
    assert PRISTINE_ANTHROPIC_MODELS == frozenset(ANTHROPIC_DEFAULT_MODELS.values())


def test_a_model_id_that_equals_a_pristine_default_is_still_remapped(
    make_settings: Callable[..., Settings]
) -> None:
    # DOCUMENTED LIMITATION, encoded as a passing test on purpose.
    # ``Settings._apply_provider_defaults`` decides "the user never configured this" by
    # *value*, not by provenance, so a user who genuinely wants
    # claude-3-5-sonnet-20241022 as the model id while running the openai provider has it
    # rewritten to gpt-4o.  The trade-off is intentional and stated in the validator's own
    # docstring ("anything that is not one of the declared Anthropic defaults is never
    # touched"); recording it here means a future switch to provenance-aware detection
    # shows up as a deliberate test change instead of a silent behaviour change.
    config = make_settings(provider=Provider.OPENAI, model="claude-3-5-sonnet-20241022")
    assert config.model == "gpt-4o"
    # The other three roles were never configured by the user, so they map as usual.
    assert config.planner_model == "gpt-4o-mini"


# --------------------------------------------------------------------------------------
# Field bounds
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("field", "bad_value"), BOUND_VIOLATIONS)
def test_bounds_are_rejected_at_construction(
    field: str, bad_value: object, make_settings: Callable[..., Settings]
) -> None:
    kwargs: dict[str, Any] = {field: bad_value}
    with pytest.raises(ValidationError):
        make_settings(**kwargs)


@pytest.mark.parametrize(("field", "bad_value"), BOUND_VIOLATIONS)
def test_bounds_are_rejected_on_assignment(
    field: str, bad_value: object, make_settings: Callable[..., Settings]
) -> None:
    config = make_settings()
    with pytest.raises(ValidationError):
        setattr(config, field, bad_value)


@pytest.mark.parametrize(("field", "bad_value"), BOUND_VIOLATIONS)
def test_a_rejected_assignment_leaves_the_previous_value_intact(
    field: str, bad_value: object, make_settings: Callable[..., Settings]
) -> None:
    before = getattr(make_settings(), field)
    config = make_settings()
    with pytest.raises(ValidationError):
        setattr(config, field, bad_value)
    assert getattr(config, field) == before


@pytest.mark.parametrize(("field", "good_value", "expected"), BOUND_ACCEPTANCES)
def test_boundary_values_are_accepted_at_construction(
    field: str,
    good_value: object,
    expected: object,
    make_settings: Callable[..., Settings],
) -> None:
    kwargs: dict[str, Any] = {field: good_value}
    assert getattr(make_settings(**kwargs), field) == expected


@pytest.mark.parametrize(("field", "good_value", "expected"), BOUND_ACCEPTANCES)
def test_boundary_values_are_accepted_on_assignment(
    field: str,
    good_value: object,
    expected: object,
    make_settings: Callable[..., Settings],
) -> None:
    config = make_settings()
    setattr(config, field, good_value)
    assert getattr(config, field) == expected


@pytest.mark.parametrize(("field", "expected"), DECLARED_BOUNDS_DEFAULTS)
def test_declared_bounds_defaults(
    field: str, expected: object, make_settings: Callable[..., Settings]
) -> None:
    assert getattr(make_settings(), field) == expected


def test_max_cycles_zero_means_unlimited(make_settings: Callable[..., Settings]) -> None:
    # max_cycles uses ge=0, not gt=0: 0 is the documented "no cap" value.
    assert make_settings(max_cycles=0).max_cycles == 0


def test_max_cycles_accepts_a_positive_cap(make_settings: Callable[..., Settings]) -> None:
    assert make_settings(max_cycles=7).max_cycles == 7


def test_unbounded_remaining_settings_defaults(make_settings: Callable[..., Settings]) -> None:
    config = make_settings()
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.log_level == "INFO"


# --------------------------------------------------------------------------------------
# API key aliases (FURROW_ prefixed and bare, from env and from .env)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        pytest.param("FURROW_ANTHROPIC_API_KEY", "prefixed-anthropic", id="prefixed"),
        pytest.param("ANTHROPIC_API_KEY", "bare-anthropic", id="bare"),
    ],
)
def test_anthropic_api_key_is_read_from_either_env_spelling(
    env_var: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Callable[..., Settings],
) -> None:
    monkeypatch.setenv(env_var, value)
    assert make_settings().anthropic_api_key == value


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        pytest.param("FURROW_OPENAI_API_KEY", "prefixed-openai", id="prefixed"),
        pytest.param("OPENAI_API_KEY", "bare-openai", id="bare"),
    ],
)
def test_openai_api_key_is_read_from_either_env_spelling(
    env_var: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Callable[..., Settings],
) -> None:
    monkeypatch.setenv(env_var, value)
    assert make_settings().openai_api_key == value


@pytest.mark.parametrize(
    ("bare_var", "prefixed_var"),
    [
        pytest.param("ANTHROPIC_API_KEY", "FURROW_ANTHROPIC_API_KEY", id="anthropic"),
        pytest.param("OPENAI_API_KEY", "FURROW_OPENAI_API_KEY", id="openai"),
    ],
)
def test_the_prefixed_env_var_wins_when_both_spellings_are_set(
    bare_var: str,
    prefixed_var: str,
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Callable[..., Settings],
) -> None:
    monkeypatch.setenv(bare_var, "bare-value")
    monkeypatch.setenv(prefixed_var, "prefixed-value")
    config = make_settings()
    field = "anthropic_api_key" if "ANTHROPIC" in bare_var else "openai_api_key"
    assert getattr(config, field) == "prefixed-value"


def test_api_keys_default_to_none_when_nothing_is_set(
    make_settings: Callable[..., Settings]
) -> None:
    config = make_settings()
    assert config.anthropic_api_key is None
    assert config.openai_api_key is None


def test_bare_anthropic_spelling_in_a_dotenv_file_is_not_ignored(tmp_path: Path) -> None:
    # Regression guard: validation_alias disables env_prefix, so without the explicit
    # AliasChoices entry the bare spelling was silently dropped from .env files too,
    # surfacing later as "ValueError: ANTHROPIC_API_KEY is not set" from the LLM client.
    env_file = _write_dotenv(tmp_path, "ANTHROPIC_API_KEY=dotenv-bare-anthropic\n")
    assert _settings_from_env_file(env_file).anthropic_api_key == "dotenv-bare-anthropic"


def test_bare_openai_spelling_in_a_dotenv_file_is_not_ignored(tmp_path: Path) -> None:
    env_file = _write_dotenv(tmp_path, "OPENAI_API_KEY=dotenv-bare-openai\n")
    assert _settings_from_env_file(env_file).openai_api_key == "dotenv-bare-openai"


def test_prefixed_spellings_in_a_dotenv_file_are_read(tmp_path: Path) -> None:
    env_file = _write_dotenv(
        tmp_path,
        "FURROW_ANTHROPIC_API_KEY=dotenv-prefixed-anthropic\nFURROW_OPENAI_API_KEY=dotenv-o\n",
    )
    config = _settings_from_env_file(env_file)
    assert config.anthropic_api_key == "dotenv-prefixed-anthropic"
    assert config.openai_api_key == "dotenv-o"


def test_the_prefixed_dotenv_spelling_wins_over_the_bare_one(tmp_path: Path) -> None:
    env_file = _write_dotenv(
        tmp_path,
        "ANTHROPIC_API_KEY=dotenv-bare\nFURROW_ANTHROPIC_API_KEY=dotenv-prefixed\n",
    )
    assert _settings_from_env_file(env_file).anthropic_api_key == "dotenv-prefixed"


# --------------------------------------------------------------------------------------
# dotenv loading
# --------------------------------------------------------------------------------------


def test_dotenv_file_drives_prefixed_settings_fields(tmp_path: Path) -> None:
    env_file = _write_dotenv(
        tmp_path,
        "FURROW_MAX_TOKENS=99\nFURROW_WORKSPACE=/tmp/furrow-ws\nFURROW_LOG_LEVEL=DEBUG\n",
    )
    config = _settings_from_env_file(env_file)
    assert config.max_tokens == 99
    assert config.workspace == Path("/tmp/furrow-ws")
    assert config.log_level == "DEBUG"


def test_dotenv_file_can_switch_the_provider_and_resolve_its_models(tmp_path: Path) -> None:
    env_file = _write_dotenv(tmp_path, "FURROW_PROVIDER=ollama\n")
    config = _settings_from_env_file(env_file)
    assert config.provider is Provider.OLLAMA
    assert config.model == "qwen2.5-coder"


def test_environment_variable_takes_precedence_over_the_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = _write_dotenv(tmp_path, "FURROW_MAX_TOKENS=99\n")
    monkeypatch.setenv("FURROW_MAX_TOKENS", "123")
    assert _settings_from_env_file(env_file).max_tokens == 123


def test_dotenv_in_the_working_directory_is_read_without_an_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_dotenv(tmp_path, "FURROW_LOG_LEVEL=DEBUG\nANTHROPIC_API_KEY=cwd-dotenv\n")
    config = Settings()
    assert config.log_level == "DEBUG"
    assert config.anthropic_api_key == "cwd-dotenv"


def test_env_file_none_opts_out_of_dotenv_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_dotenv(tmp_path, "FURROW_LOG_LEVEL=DEBUG\n")
    assert _settings_from_env_file(None).log_level == "INFO"


def test_a_missing_dotenv_file_is_not_an_error(make_settings: Callable[..., Settings]) -> None:
    assert make_settings(workspace=Path("/tmp/nowhere-at-all")).workspace == Path(
        "/tmp/nowhere-at-all"
    )


# --------------------------------------------------------------------------------------
# workspace default factory
# --------------------------------------------------------------------------------------


def test_workspace_defaults_to_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _settings_from_env_file(None).workspace == tmp_path


def test_workspace_is_coerced_to_a_path(make_settings: Callable[..., Settings]) -> None:
    assert make_settings(workspace="/tmp/furrow-ws").workspace == Path("/tmp/furrow-ws")


def test_workspace_defaults_are_not_shared_between_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _settings_from_env_file(None).workspace is not _settings_from_env_file(None).workspace


# --------------------------------------------------------------------------------------
# module level singleton
# --------------------------------------------------------------------------------------


def test_module_level_settings_singleton_is_a_materialised_settings() -> None:
    assert isinstance(settings, Settings)
    assert settings.provider in set(Provider)
    # Dumping proves the singleton is fully built rather than a half-initialised shell.
    assert "provider" in settings.model_dump()