from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_result_is_optional():
    """TaskModel.result should be Optional[str] (str | None)."""
    task = TaskModel(id="1", description="test")
    assert task.result is None
    task2 = TaskModel(id="2", description="test2", result="some output")
    assert task2.result == "some output"