from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="do thing")
    assert task.status == "pending"
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_test_result_failures_default():
    t = TestResult(passed=False, summary="broken")
    assert t.failures == []


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []
