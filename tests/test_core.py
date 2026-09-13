from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.id == "1"
    assert t.description == "test"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="no tasks")
    assert p.tasks == []
    assert p.rationale == "no tasks"


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="2 failures", failures=["error1", "error2"])
    assert t.passed is False
    assert t.summary == "2 failures"
    assert t.failures == ["error1", "error2"]
    assert len(t.failures) == 2
