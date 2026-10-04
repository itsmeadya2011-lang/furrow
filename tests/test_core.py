"""Cross-cutting invariants for the shared foundation: enums, models and settings.

These tests deliberately reach across modules: every `EventType` must stay a distinct
string, every `Settings` field must stay reachable from its `FURROW_`-prefixed environment
variable, and every model class must survive a `model_dump()` / `model_validate()` round trip.
A new field or enum member that breaks one of those contracts fails here first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    get_settings,
    reload_settings,
)
from furrow.config import (
    settings as singleton,
)
from furrow.events import Event, EventBus, EventType

PATH_MARKER = "<tmp path>"

# (field name, raw env string, expected parsed value). PATH_MARKER is replaced with a real
# path inside tmp_path, so no test ever points a setting at the repository.
ENV_CASES: list[tuple[str, str, Any]] = [
    ("provider", "ollama", Provider.OLLAMA),
    ("model", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
    ("planner_model", "claude-3-5-haiku-20241022", "claude-3-5-haiku-20241022"),
    ("worker_model", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
    ("tester_model", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
    ("anthropic_api_key", "sk-anthropic", "sk-anthropic"),
    ("openai_api_key", "sk-openai", "sk-openai"),
    ("ollama_base_url", "http://ollama.internal:11434", "http://ollama.internal:11434"),
    ("max_parallel_tasks", "9", 9),
    ("max_cycles", "4", 4),
    ("workspace", PATH_MARKER, Path),
    ("log_level", "DEBUG", "DEBUG"),
    ("temperature", "0.25", 0.25),
    ("max_tokens", "1234", 1234),
    ("max_retries", "7", 7),
    ("retry_base_delay", "0.5", 0.5),
    ("request_timeout", "12.5", 12.5),
    ("test_command", "python3 -m pytest -q", "python3 -m pytest -q"),
    ("test_timeout", "45", 45),
    ("lint_command", "python3 -m ruff check .", "python3 -m ruff check ."),
    ("typecheck_command", "python3 -m mypy .", "python3 -m mypy ."),
    ("exclude_dirs", '["cache", "*.tmp"]', ["cache", "*.tmp"]),
    ("web_host", "0.0.0.0", "0.0.0.0"),
    ("web_port", "9001", 9001),
    ("state_file", PATH_MARKER, Path),
    ("max_tool_iterations", "5", 5),
    ("max_stalled_cycles", "6", 6),
    ("max_event_history", "11", 11),
]


@pytest.mark.parametrize(("field", "raw", "expected"), ENV_CASES)
def test_every_settings_field_is_settable_from_the_environment(
    field: str, raw: str, expected: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wanted = tmp_path / "value" if raw == PATH_MARKER else raw
    monkeypatch.setenv(f"FURROW_{field.upper()}", str(wanted))
    parsed = reload_settings()
    got = getattr(parsed, field)
    assert got == (wanted if raw == PATH_MARKER else expected)
    assert type(got) is type(parsed.model_dump()[field])


def test_the_env_case_table_covers_every_settings_field() -> None:
    assert {name for name, _, _ in ENV_CASES} == set(Settings.model_fields)


# ------------------------------------------------------------------- enum values
def test_every_event_type_has_a_distinct_string_value() -> None:
    values = [member.value for member in EventType]
    assert len(values) == len(set(values))
    assert all(isinstance(value, str) and value for value in values)
    assert all(value == value.lower() for value in values)
    assert all(" " not in value for value in values)


def test_every_event_type_value_mirrors_its_name() -> None:
    # Event values travel through URLs and websocket frames: keep them boring.
    for member in EventType:
        assert member.value == member.name.lower()
        assert member.value.replace("_", "").isalnum()


def test_the_task_status_and_provider_enums_do_not_overlap() -> None:
    assert not {member.value for member in TaskStatus} & {member.value for member in Provider}


# ------------------------------------------------------------------ model shape
def test_plan_construction_is_meaningful() -> None:
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", files=["a.py"], dependencies=["2"])],
        rationale="ok",
    )
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert (task.id, task.description, task.files, task.dependencies) == (
        "1",
        "do thing",
        ["a.py"],
        ["2"],
    )
    assert plan.rationale == "ok"
    assert task.status is TaskStatus.PENDING
    assert task.attempts == 0


def test_test_result_construction_is_meaningful() -> None:
    result = TestResult(passed=True, summary="ok", failures=[])
    assert result.passed is True
    assert result.summary == "ok"
    assert result.failures == []
    assert result.commands == []


@pytest.mark.parametrize(
    "model",
    [
        TaskModel(id="1", description="d", files=["a.py"], dependencies=["2"]),
        Plan(tasks=[TaskModel(id="1", description="d")], rationale="r"),
        TestResult(passed=False, summary="s", failures=["f"], commands=["pytest -q"]),
    ],
)
def test_models_round_trip_through_dump_and_validate(model: Any) -> None:
    payload = model.model_dump()
    assert type(model).model_validate(payload) == model
    assert json_round_trip(payload) == payload


def test_task_status_round_trips_as_its_string_value() -> None:
    payload = TaskModel(id="1", description="d", status=TaskStatus.FAILED).model_dump()
    assert payload["status"] == "failed"
    assert TaskModel.model_validate(payload).status is TaskStatus.FAILED


def test_the_settings_singleton_is_reachable_and_not_mutated() -> None:
    assert get_settings() is singleton
    assert get_settings() is get_settings()
    assert isinstance(get_settings(), Settings)
    assert reload_settings() is not singleton


# ------------------------------------------------------------------- event bus
def test_an_event_bus_round_trips_events_through_dicts() -> None:
    bus = EventBus(history=5)
    bus.emit(EventType.TASK_END, "task 1 failed", cycle=2, id="1", status="failed")
    payloads = [event.to_dict() for event in bus.history()]
    assert json_round_trip(payloads) == payloads
    assert payloads[0]["data"] == {"id": "1", "status": "failed"}


def test_event_bus_defaults_keep_a_short_history() -> None:
    bus = EventBus()
    for index in range(600):
        bus.emit(EventType.LOG, str(index))
    assert len(bus.history()) == 500
    assert bus.history()[-1].message == "599"


def json_round_trip(value: Any) -> Any:
    """Serialise through JSON so a test proves a payload is transport-safe."""
    return json.loads(json.dumps(value))


def test_event_dataclass_defaults_survive_construction() -> None:
    event = Event(type=EventType.LOG)
    assert Event(**{**event.to_dict(), "type": EventType.LOG}).message == event.message
