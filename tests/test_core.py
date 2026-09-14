from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_plan_multiple_tasks():
    """Plan model with multiple tasks."""
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
        TaskModel(id="3", description="task 3"),
    ]
    p = Plan(tasks=tasks, rationale="multi-task plan")
    assert len(p.tasks) == 3
    assert p.tasks[0].id == "1"
    assert p.tasks[1].description == "task 2"
    assert p.rationale == "multi-task plan"


def test_test_result_with_failures():
    """TestResult with failures list."""
    failures = ["test_a failed", "test_b failed"]
    t = TestResult(passed=False, summary="2 failures", failures=failures)
    assert t.passed is False
    assert t.summary == "2 failures"
    assert len(t.failures) == 2
    assert "test_a failed" in t.failures


def test_task_model_with_dependencies():
    """TaskModel with dependencies and files."""
    t = TaskModel(
        id="task-1",
        description="implement feature",
        files=["src/feature.py", "tests/test_feature.py"],
        dependencies=["task-0"],
        status="pending",
    )
    assert t.id == "task-1"
    assert len(t.files) == 2
    assert "src/feature.py" in t.files
    assert len(t.dependencies) == 1
    assert t.dependencies[0] == "task-0"
    assert t.status == "pending"
    assert t.result is None
