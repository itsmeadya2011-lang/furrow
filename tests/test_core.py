from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.cycle_timeout == 300


def test_plan_with_dependencies():
    tasks = [
        TaskModel(id="1", description="task 1", dependencies=[]),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
    ]
    p = Plan(tasks=tasks, rationale="test")
    assert p.tasks[1].dependencies == ["1"]


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="failed", failures=["test1", "test2"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "3")
    s = Settings()
    assert s.max_cycles == 10
    assert s.max_parallel_tasks == 3