import json

from furrow.config import Plan, TaskModel, TestResult


def test_task_model_defaults():
    task = TaskModel(id="t1", description="do something")
    assert task.id == "t1"
    assert task.description == "do something"
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_all_fields():
    task = TaskModel(
        id="t2",
        description="complex task",
        files=["a.py", "b.py"],
        dependencies=["t1"],
        status="in_progress",
        result="partial",
    )
    assert task.files == ["a.py", "b.py"]
    assert task.dependencies == ["t1"]
    assert task.status == "in_progress"
    assert task.result == "partial"


def test_plan_with_tasks():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="first"),
            TaskModel(id="2", description="second"),
        ],
        rationale="build the thing",
    )
    assert len(plan.tasks) == 2
    assert plan.rationale == "build the thing"
    assert plan.tasks[0].id == "1"


def test_test_result_construction():
    tr = TestResult(passed=True, summary="all good", failures=[])
    assert tr.passed is True
    assert tr.summary == "all good"
    assert tr.failures == []


def test_test_result_with_failures():
    tr = TestResult(
        passed=False,
        summary="3 failed",
        failures=["test_a", "test_b", "test_c"],
    )
    assert tr.passed is False
    assert tr.failures == ["test_a", "test_b", "test_c"]


def test_task_model_json_roundtrip():
    task = TaskModel(
        id="t1", description="hello", files=["x.py"], dependencies=["t0"]
    )
    raw = task.model_dump_json()
    restored = TaskModel.model_validate_json(raw)
    assert restored == task


def test_plan_json_roundtrip():
    plan = Plan(
        tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")],
        rationale="because",
    )
    raw = plan.model_dump_json()
    restored = Plan.model_validate_json(raw)
    assert restored.tasks[0].id == "1"
    assert restored.tasks[1].description == "b"
    assert restored.rationale == "because"


def test_test_result_json_roundtrip():
    tr = TestResult(passed=False, summary="x", failures=["a", "b"])
    raw = tr.model_dump_json()
    restored = TestResult.model_validate_json(raw)
    assert restored.passed is False
    assert restored.failures == ["a", "b"]


def test_plan_from_valid_json():
    payload = {
        "tasks": [
            {"id": "1", "description": "alpha"},
            {"id": "2", "description": "beta", "files": ["f.py"]},
        ],
        "rationale": "we should do it",
    }
    plan = Plan.model_validate(json.dumps(payload))
    assert len(plan.tasks) == 2
    assert plan.tasks[1].files == ["f.py"]
    assert plan.rationale == "we should do it"