"""Contract tests for :mod:`furrow.models`, the pure pydantic schemas.

Most assertions here look trivial, and that is the point: they pin the *defaults* and the
*permissiveness* that the rest of furrow is written against. Concretely,

* ``TaskModel.status`` defaults to :attr:`TaskStatus.PENDING` and ``result`` to ``None``,
  because a freshly planned task has not run;
* ``Plan.rationale`` and ``TestResult.passed`` are optional. They are produced by an LLM,
  and a model that omits a narrative field should not invalidate an otherwise usable plan.
  Missing ``passed`` degrades to the conservative ``False``;
* ``TaskModel`` carries **no** ``model_config``, so assignment is not validated. That is
  load-bearing, not an oversight -- see :func:`test_assignment_is_not_validated`.

Nothing in this module imports ``furrow.llm``, ``furrow.agents`` or a provider SDK.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from furrow import models
from furrow.models import (
    FileChange,
    Plan,
    TaskModel,
    TaskStatus,
    TestResult,
    WorkerResult,
)


def test_task_status_has_exactly_the_five_documented_string_values():
    assert [status.value for status in TaskStatus] == [
        "pending",
        "running",
        "completed",
        "failed",
        "skipped",
    ]


def test_task_status_members_are_strings():
    # The `str` mixin is what makes `task.status == "completed"` (used in
    # core/orchestrator.py and agents/worker.py) work at all.
    assert isinstance(TaskStatus.PENDING, str)
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus("failed") is TaskStatus.FAILED


def test_task_model_defaults():
    task = TaskModel(id="1", description="add add() to app.py")

    assert task.id == "1"
    assert task.description == "add add() to app.py"
    assert task.status is TaskStatus.PENDING
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_task_model_requires_an_id_and_a_description():
    # The only two identifying fields are mandatory; everything else has a default.
    with pytest.raises(ValidationError) as excinfo:
        TaskModel()  # type: ignore[call-arg]
    errors = {error["loc"][0] for error in excinfo.value.errors()}
    assert errors == {"id", "description"}

    with pytest.raises(ValidationError):
        TaskModel(id="1")  # type: ignore[call-arg]


def test_task_model_list_defaults_are_not_shared_between_instances():
    # `Field(default_factory=list)`, not a shared `[]`: mutating one task's file list must
    # never be visible from another task, or tasks would leak scope into each other.
    first = TaskModel(id="1", description="a")
    second = TaskModel(id="2", description="b")
    first.files.append("app.py")
    first.dependencies.append("2")

    assert second.files == []
    assert second.dependencies == []


def test_plan_defaults():
    plan = Plan()

    assert plan.tasks == []
    assert plan.rationale == ""


def test_plan_tasks_default_is_not_shared_between_instances():
    first, second = Plan(), Plan()
    first.tasks.append(TaskModel(id="1", description="a"))
    assert second.tasks == []


def test_plan_accepts_a_verbose_model_response():
    plan = Plan(
        tasks=[TaskModel(id="1", description="do it", files=["a.py"], dependencies=[])],
        rationale="one slice",
    )

    assert len(plan.tasks) == 1
    assert plan.tasks[0].status is TaskStatus.PENDING
    assert plan.rationale == "one slice"


def test_test_result_defaults():
    # A model that omits `passed` must not be read as a pass, hence False and not True.
    result = TestResult()

    assert result.passed is False
    assert result.summary == ""
    assert result.failures == []


def test_test_result_skipped_defaults_false():
    result = TestResult()

    assert result.skipped is False
    # `skipped` is a distinct concept from `passed`: "no suite to run" is neutral, not a
    # pass and not a regression, so it must not imply either.
    assert TestResult(skipped=True).passed is False
    assert TestResult(passed=True).skipped is False


def test_file_change_action_defaults_to_write():
    change = FileChange(path="a.py", content="x = 1\n")

    assert change.action == "write"


def test_file_change_accepts_an_explicit_delete():
    change = FileChange(path="a.py", content="", action="delete")

    assert change.action == "delete"


def test_worker_result_defaults():
    result = WorkerResult()

    assert result.summary == ""
    assert result.files == []
    assert result.notes == ""


def test_worker_result_files_hold_file_changes():
    result = WorkerResult(
        summary="done",
        files=[FileChange(path="a.py", content="x = 1\n"), FileChange(path="b.py", content="")],
    )

    assert [change.path for change in result.files] == ["a.py", "b.py"]
    assert [change.action for change in result.files] == ["write", "write"]


def test_a_raw_string_status_compares_equal_to_the_enum_member():
    # The orchestrator assigns plain strings onto `status`; because TaskStatus is a `str`
    # enum the comparison below is True either way.
    task = TaskModel(id="1", description="d", status="completed")  # type: ignore[arg-type]

    assert task.status == "completed"
    assert task.status == TaskStatus.COMPLETED
    assert TaskModel(id="1", description="d").status == "pending"


def test_assignment_is_not_validated():
    """`TaskModel` deliberately has no `model_config`, so `validate_assignment` is off.

    The orchestrator mutates a live task while a cycle is in flight -- including assigning
    a raw string that is not a `TaskStatus` member, and assigning a non-string payload into
    `result`. With `validate_assignment=True` those mid-run assignments would raise and kill
    the cycle. Asserted here so a future "let's be strict" cleanup breaks loudly rather
    than silently.
    """
    task = TaskModel(id="1", description="d")

    task.status = "blocked"  # type: ignore[assignment]
    task.result = {"summary": "structured, not a string"}  # type: ignore[assignment]

    assert task.status == "blocked"
    assert task.status not in set(TaskStatus)
    assert task.result == {"summary": "structured, not a string"}


def test_model_dump_json_mode_renders_enums_as_plain_strings():
    # The event stream and the CLI serialise `model_dump()`; python mode keeps the enum
    # member (fine in-process, but not JSON) while `mode="json"` yields plain strings.
    task = TaskModel(id="1", description="d", status=TaskStatus.COMPLETED)

    assert task.model_dump()["status"] is TaskStatus.COMPLETED

    dumped = task.model_dump(mode="json")
    assert dumped["status"] == "completed"
    assert type(dumped["status"]) is str
    assert json.loads(task.model_dump_json())["status"] == "completed"


def test_models_module_exports_exactly_the_documented_surface():
    assert models.__all__ == [
        "FileChange",
        "Plan",
        "TaskModel",
        "TaskStatus",
        "TestResult",
        "WorkerResult",
    ]
    for name in models.__all__:
        assert getattr(models, name) is not None


def test_test_result_is_marked_uncollectable():
    """``tests/conftest.py`` must keep the `TestResult` collection warning out of the run.

    ``TestResult`` matches pytest's default ``Test*`` class pattern, so without
    ``__test__ = False`` every run prints
    ``PytestCollectionWarning: cannot collect test class 'TestResult'``. ``pyproject.toml``
    is not ours to edit, so the marker is set from ``pytest_configure``.
    """
    assert TestResult.__test__ is False  # type: ignore[attr-defined]
