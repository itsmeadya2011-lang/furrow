from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    """TaskModel should default status to pending and empty files/dependencies."""
    t = TaskModel(id="1", description="test")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_test_result_defaults():
    """TestResult should default failures to empty list."""
    t = TestResult(passed=True, summary="ok")
    assert t.failures == []


def test_plan_empty_tasks():
    """Plan should accept an empty tasks list."""
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []