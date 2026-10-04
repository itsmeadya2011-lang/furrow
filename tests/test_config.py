"""Tests for :mod:`furrow.config`.

These tests are fast, fully offline and deterministic:

* no network access is performed;
* no ``.env`` file is required -- every :class:`Settings` instance under test is
  built with ``_env_file=None`` so a stray ``.env`` on disk cannot change the
  outcome;
* module level import-time state (``furrow.config.settings``) is never relied
  upon, so a developer's real environment cannot leak into assertions;
* every ``FURROW_*`` environment variable is scrubbed before each test.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TaskStatus, TestResult

# Every ``FURROW_*`` variable ``Settings`` knows about, mapped to the value used
# by the env-var test and the attribute/expected result it must produce.
ENV_CASES = [
    ("FURROW_PROVIDER", "ollama", "provider", Provider.OLLAMA),
    ("FURROW_MODEL", "gpt-4o", "model", "gpt-4o"),
    ("FURROW_PLANNER_MODEL", "planner-model", "planner_model", "planner-model"),
    ("FURROW_WORKER_MODEL", "worker-model", "worker_model", "worker-model"),
    ("FURROW_TESTER_MODEL", "tester-model", "tester_model", "tester-model"),
    ("FURROW_ANTHROPIC_API_KEY", "sk-ant-xyz", "anthropic_api_key", "sk-ant-xyz"),
    ("FURROW_OPENAI_API_KEY", "sk-oai-xyz", "openai_api_key", "sk-oai-xyz"),
    (
        "FURROW_OLLAMA_BASE_URL",
        "http://ollama.internal:11434",
        "ollama_base_url",
        "http://ollama.internal:11434",
    ),
    ("FURROW_MAX_PARALLEL_TASKS", "9", "max_parallel_tasks", 9),
    ("FURROW_MAX_CYCLES", "7", "max_cycles", 7),
    ("FURROW_WORKSPACE", "/tmp/furrow-ws", "workspace", Path("/tmp/furrow-ws")),
    ("FURROW_LOG_LEVEL", "DEBUG", "log_level", "DEBUG"),
]


@pytest.fixture(autouse=True)
def _scrub_furrow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ``FURROW_*`` variable so the ambient environment is irrelevant."""
    for name in [n for n in os.environ if n.upper().startswith("FURROW_")]:
        monkeypatch.delenv(name, raising=False)


def make_settings() -> Settings:
    """Build a ``Settings`` instance that ignores any ``.env`` file on disk."""
    return Settings(_env_file=None)


# --------------------------------------------------------------------------- #
# env prefix / env file contract
# --------------------------------------------------------------------------- #


def test_settings_uses_furrow_env_prefix_and_dotenv() -> None:
    assert Settings.model_config["env_prefix"] == "FURROW_"
    assert Settings.model_config["env_file"] == ".env"


# --------------------------------------------------------------------------- #
# 1. every FURROW_* env var is honoured
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("env_var", "raw", "attr", "expected"), ENV_CASES)
def test_env_var_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    raw: str,
    attr: str,
    expected: object,
) -> None:
    monkeypatch.setenv(env_var, raw)
    settings = make_settings()
    assert getattr(settings, attr) == expected


def test_all_documented_env_vars_are_covered() -> None:
    """Every ``Settings`` field is reachable through a ``FURROW_`` env var."""
    covered = {attr for _, _, attr, _ in ENV_CASES}
    assert covered == set(Settings.model_fields)


def test_int_env_vars_are_coerced_from_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "12")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "3")
    settings = make_settings()
    assert settings.max_parallel_tasks == 12
    assert settings.max_cycles == 3


def test_provider_env_var_accepts_each_member(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw, expected in (("anthropic", Provider.ANTHROPIC), ("openai", Provider.OPENAI)):
        monkeypatch.setenv("FURROW_PROVIDER", raw)
        assert make_settings().provider is expected


def test_provider_env_var_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "definitely-not-a-provider")
    with pytest.raises(ValidationError):
        make_settings()


# --------------------------------------------------------------------------- #
# 2. documented defaults
# --------------------------------------------------------------------------- #


def test_settings_defaults() -> None:
    settings = make_settings()
    assert settings.provider is Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.planner_model == "claude-3-5-haiku-20241022"
    assert settings.worker_model == "claude-3-5-sonnet-20241022"
    assert settings.tester_model == "claude-3-5-sonnet-20241022"
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.workspace == Path.cwd()
    assert settings.log_level == "INFO"


def test_max_cycles_zero_means_unlimited() -> None:
    """``max_cycles == 0`` is the documented "no cycle limit" sentinel."""
    assert make_settings().max_cycles == 0


def test_workspace_default_is_a_path() -> None:
    assert isinstance(make_settings().workspace, Path)


def test_keyword_overrides_beat_defaults() -> None:
    settings = Settings(_env_file=None, max_parallel_tasks=2, log_level="WARNING")
    assert settings.max_parallel_tasks == 2
    assert settings.log_level == "WARNING"


def test_explicit_kwargs_beat_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MODEL", "from-env")
    assert Settings(_env_file=None, model="from-kwargs").model == "from-kwargs"


# --------------------------------------------------------------------------- #
# 3. TaskStatus literal validation
# --------------------------------------------------------------------------- #


def test_task_status_literal_members() -> None:
    assert get_args(TaskStatus) == ("pending", "in_progress", "completed", "failed")


def test_task_model_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="1", description="d", status="bogus")


@pytest.mark.parametrize("status", ["pending", "in_progress", "completed", "failed"])
def test_task_model_accepts_every_status(status: str) -> None:
    assert TaskModel(id="1", description="d", status=status).status == status


def test_task_status_defaults_to_pending() -> None:
    assert TaskModel(id="1", description="d").status == "pending"


def test_task_status_is_a_plain_str_literal() -> None:
    """Status stays a plain ``str`` so callers may assign raw literals."""
    task = TaskModel(id="1", description="d")
    task.status = "failed"
    assert task.status == "failed"
    assert isinstance(task.status, str)
    assert not isinstance(task.status, Enum)


def test_task_model_other_defaults() -> None:
    task = TaskModel(id="1", description="d")
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_task_model_mutable_defaults_are_not_shared() -> None:
    first = TaskModel(id="1", description="a")
    second = TaskModel(id="2", description="b")
    first.files.append("x.py")
    assert first.files == ["x.py"]
    assert second.files == []


# --------------------------------------------------------------------------- #
# 4. TestResult must not be collected by pytest
# --------------------------------------------------------------------------- #


def test_test_result_is_excluded_from_collection() -> None:
    assert TestResult.__test__ is False


def test_test_result_is_still_a_usable_pydantic_model() -> None:
    result = TestResult(passed=True, summary="all green")
    assert result.passed is True
    assert result.summary == "all green"
    assert result.failures == []
    assert set(TestResult.model_fields) == {"passed", "summary", "failures"}


def test_test_result_requires_its_mandatory_fields() -> None:
    with pytest.raises(ValidationError):
        TestResult(passed=True)  # type: ignore[call-arg]


def test_test_result_failures_default_is_not_shared() -> None:
    first = TestResult(passed=False, summary="a")
    second = TestResult(passed=False, summary="b")
    first.failures.append("boom")
    assert first.failures == ["boom"]
    assert second.failures == []


# --------------------------------------------------------------------------- #
# 5. round-trips through model_dump / model_validate, JSON serialisable
# --------------------------------------------------------------------------- #


def test_task_model_round_trip() -> None:
    original = TaskModel(
        id="42",
        description="implement the thing",
        files=["a.py", "b.py"],
        dependencies=["41"],
        status="completed",
        result="done",
    )
    dumped = original.model_dump()
    assert dumped == {
        "id": "42",
        "description": "implement the thing",
        "files": ["a.py", "b.py"],
        "dependencies": ["41"],
        "status": "completed",
        "result": "done",
    }
    assert TaskModel.model_validate(dumped) == original


def test_task_model_dump_is_json_serializable() -> None:
    dumped = TaskModel(id="1", description="d").model_dump()
    assert json.loads(json.dumps(dumped)) == dumped


def test_plan_round_trip() -> None:
    original = Plan(
        tasks=[
            TaskModel(id="1", description="first"),
            TaskModel(id="2", description="second", dependencies=["1"]),
        ],
        rationale="because",
    )
    dumped = original.model_dump()
    assert Plan.model_validate(dumped) == original
    assert json.loads(json.dumps(dumped)) == dumped


def test_test_result_round_trip() -> None:
    original = TestResult(passed=False, summary="2 failed", failures=["t1", "t2"])
    dumped = original.model_dump()
    assert dumped == {"passed": False, "summary": "2 failed", "failures": ["t1", "t2"]}
    assert TestResult.model_validate(dumped) == original
    assert json.loads(json.dumps(dumped)) == dumped


def test_models_support_json_mode_dump() -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="d")], rationale="r")
    assert json.loads(plan.model_dump_json()) == plan.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# 6. Provider enum
# --------------------------------------------------------------------------- #


def test_provider_members_and_values() -> None:
    assert [member.name for member in Provider] == ["ANTHROPIC", "OPENAI", "OLLAMA"]
    assert [member.value for member in Provider] == ["anthropic", "openai", "ollama"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("anthropic", Provider.ANTHROPIC),
        ("openai", Provider.OPENAI),
        ("ollama", Provider.OLLAMA),
    ],
)
def test_provider_constructs_from_value(raw: str, expected: Provider) -> None:
    assert Provider(raw) is expected


def test_provider_is_a_str_enum() -> None:
    assert Provider.OLLAMA == "ollama"
    assert isinstance(Provider.OLLAMA, str)


def test_provider_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        Provider("bedrock")


# --------------------------------------------------------------------------- #
# Plan validation
# --------------------------------------------------------------------------- #


def test_plan_requires_tasks_and_rationale() -> None:
    with pytest.raises(ValidationError):
        Plan(tasks=[])  # type: ignore[call-arg]


def test_plan_rejects_non_task_items() -> None:
    with pytest.raises(ValidationError):
        Plan(tasks=["not-a-task"], rationale="r")  # type: ignore[list-item]
