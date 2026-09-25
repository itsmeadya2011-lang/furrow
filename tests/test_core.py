import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model():
    t = TaskModel(id="1", description="test task", files=["a.py"], dependencies=["2"])
    assert t.id == "1"
    assert t.files == ["a.py"]
    assert t.dependencies == ["2"]
    assert t.status == "pending"


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
    ]
    p = Plan(tasks=tasks, rationale="two tasks")
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])