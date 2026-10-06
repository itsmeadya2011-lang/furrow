from furrow.config import Plan, TaskModel, TestResult, Settings


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="implement feature")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_settings_defaults():
    s = Settings()
    assert s.max_cycles == 0
    assert s.max_tokens == 4096
    assert s.max_parallel_tasks == 5


def test_settings_max_cycles_override():
    s = Settings(max_cycles=3, max_tokens=2048)
    assert s.max_cycles == 3
    assert s.max_tokens == 2048
