import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_full():
    t = TaskModel(
        id="1",
        description="implement auth",
        files=["src/auth.py", "tests/test_auth.py"],
        dependencies=["2"],
        status="pending",
        result=None,
    )
    assert t.id == "1"
    assert t.files == ["src/auth.py", "tests/test_auth.py"]
    assert t.dependencies == ["2"]
    assert t.status == "pending"
    assert t.result is None


def test_plan_model_dump():
    p = Plan(
        tasks=[TaskModel(id="1", description="do thing", files=["a.py"], dependencies=[])],
        rationale="test",
    )
    dump = p.model_dump()
    assert dump["rationale"] == "test"
    assert len(dump["tasks"]) == 1
    assert dump["tasks"][0]["id"] == "1"


def test_test_result_with_failures():
    t = TestResult(
        passed=False,
        summary="1 test failed",
        failures=["AssertionError: expected True, got False"],
    )
    assert t.passed is False
    assert len(t.failures) == 1
    assert "AssertionError" in t.failures[0]


def test_test_result_rejects_invalid_passed_type():
    with pytest.raises(Exception):
        TestResult(passed="yes", summary="bad type", failures=[])
