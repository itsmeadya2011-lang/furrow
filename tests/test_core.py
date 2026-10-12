import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="2 failures", failures=["fail1", "fail2"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []
    assert p.rationale == "nothing to do"


def test_task_model_with_files_and_deps():
    t = TaskModel(
        id="2",
        description="implement feature",
        files=["src/a.py", "tests/test_a.py"],
        dependencies=["1"],
    )
    assert t.files == ["src/a.py", "tests/test_a.py"]
    assert t.dependencies == ["1"]


def test_test_result_passed_is_bool():
    t = TestResult(passed=True, summary="ok")
    assert isinstance(t.passed, bool)
    assert t.passed is True