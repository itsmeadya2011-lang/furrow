"""Unit tests for the data models in ``furrow.config`` (``Plan``/``TaskModel``/
``TestResult``) and the ``Provider`` enum.

Protects the payloads that travel between the orchestrator, the agents and the
LLM: field defaults, JSON round-tripping, and -- importantly -- that
``TaskModel.id`` is a strict string. Pydantic 2 rejects ``id=1`` instead of
coercing it, so a model that emits numeric ids currently fails validation inside
the planner, where the error is swallowed into an ERROR event instead of being
reported. Guarding that keeps the failure visible in tests.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import furrow.config as config_mod
from furrow.config import Plan, Provider, TaskModel


def _sample_plan() -> Plan:
    return Plan(
        tasks=[
            TaskModel(id="1", description="Add auth", files=["src/auth.py"]),
            TaskModel(id="2", description="Add tests", dependencies=["1"]),
        ],
        rationale="two steps",
    )


def test_task_model_defaults() -> None:
    task = TaskModel(id="1", description="do thing")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_mutable_defaults_are_not_shared() -> None:
    first, second = TaskModel(id="1", description="a"), TaskModel(id="2", description="b")
    first.files.append("x")
    first.dependencies.append("y")
    assert (second.files, second.dependencies) == ([], [])


def test_task_model_rejects_a_numeric_id() -> None:
    # pydantic 2 does not coerce int -> str; an int id from the model raises here.
    with pytest.raises(ValidationError) as excinfo:
        TaskModel(id=1, description="numeric id")
    assert "id" in str(excinfo.value)


def test_plan_with_numeric_task_ids_fails_validation() -> None:
    # The exact payload shape the planner receives from a model that numbers its
    # tasks 1, 2, 3 instead of "1", "2", "3".
    payload = {"tasks": [{"id": 1, "description": "first"}], "rationale": "r"}
    with pytest.raises(ValidationError):
        Plan(**payload)  # type: ignore[arg-type]


def test_plan_requires_tasks_and_rationale() -> None:
    with pytest.raises(ValidationError):
        Plan(tasks=[TaskModel(id="1", description="a")])  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Plan(tasks=[])  # type: ignore[call-arg]


def test_plan_round_trips_through_json() -> None:
    plan = _sample_plan()
    restored = Plan.model_validate_json(plan.model_dump_json())
    assert restored == plan
    assert restored.tasks[1].dependencies == ["1"]


def test_plan_validates_a_payload_from_the_model() -> None:
    plan = Plan.model_validate(
        {
            "tasks": [
                {
                    "id": "1",
                    "description": "Add auth",
                    "files": ["src/auth.py"],
                    "dependencies": [],
                    "status": "pending",
                    "result": None,
                }
            ],
            "rationale": "one step",
        }
    )
    assert isinstance(plan.tasks[0], TaskModel)
    assert plan.tasks[0].files == ["src/auth.py"]


def test_plan_model_copy_rewrites_fields_without_sharing_nested_models() -> None:
    # `_normalize_plan` relies on this: it copies each task before rewriting ids.
    plan = _sample_plan()
    copy = plan.model_copy(update={"rationale": "changed"})
    assert copy.rationale == "changed"
    assert plan.rationale == "two steps"
    # A shallow copy shares the nested task objects, which is exactly why the planner
    # rebuilds the task list instead of mutating the plan in place.
    assert copy.tasks[0] is plan.tasks[0]


def test_deep_model_copy_is_independent() -> None:
    plan = _sample_plan()
    copy = plan.model_copy(deep=True)
    copy.tasks[0].files.append("extra")
    copy.rationale = "changed"
    assert plan.tasks[0].files == ["src/auth.py"]
    assert plan.rationale == "two steps"


def test_task_model_model_copy_rewrites_fields_without_touching_the_original() -> None:
    task = TaskModel(id="1", description="  spaced  ", dependencies=["1"])
    copy = task.model_copy(update={"id": "2", "description": "spaced"})
    assert (copy.id, copy.description) == ("2", "spaced")
    assert copy.dependencies == ["1"]
    assert (task.id, task.description) == ("1", "  spaced  ")


def test_task_model_round_trips_through_json() -> None:
    task = TaskModel(id="7", description="Ship", files=["a.py"], status="done", result="ok")
    assert TaskModel.model_validate_json(task.model_dump_json()) == task


def test_task_model_rejects_a_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="1")  # type: ignore[call-arg]


def test_test_result_defaults_failures_to_an_empty_list() -> None:
    result = config_mod.TestResult(passed=True, summary="ok")
    assert result.failures == []
    assert result.passed is True


def test_test_result_round_trips_through_json() -> None:
    result = config_mod.TestResult(passed=False, summary="2 failed", failures=["a", "b"])
    assert config_mod.TestResult.model_validate_json(result.model_dump_json()) == result


def test_test_result_requires_a_verdict_and_summary() -> None:
    with pytest.raises(ValidationError):
        config_mod.TestResult(summary="missing verdict")  # type: ignore[call-arg]


def test_provider_is_a_string_enum() -> None:
    assert issubclass(Provider, str)
    assert Provider.OLLAMA == "ollama"
    assert {member.value for member in Provider} == {"anthropic", "openai", "ollama"}


def test_provider_survives_a_json_round_trip() -> None:
    settings_field = {"provider": Provider.ANTHROPIC}
    assert Provider(settings_field["provider"].value) is Provider.ANTHROPIC
    assert Provider("ollama") is Provider.OLLAMA