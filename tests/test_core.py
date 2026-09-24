from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test task")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_files():
    task = TaskModel(
        id="1", description="test task", files=["src/main.py"], dependencies=["task-0"]
    )
    assert task.files == ["src/main.py"]
    assert task.dependencies == ["task-0"]


def test_plan_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    p = Plan(tasks=tasks, rationale="two tasks")
    assert len(p.tasks) == 2
    assert p.tasks[0].id == "1"
    assert p.tasks[1].id == "2"


def test_test_result_with_failures():
    t = TestResult(
        passed=False, summary="tests failed", failures=["test_foo failed", "test_bar failed"]
    )
    assert t.passed is False
    assert len(t.failures) == 2
    assert "test_foo failed" in t.failures
