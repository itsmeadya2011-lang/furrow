from furrow.config import Plan, Settings, TaskModel, TestResult


def test_plan_validation():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "ok"


def test_task_model_defaults():
    task = TaskModel(id="1", description="task")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_test_result_defaults():
    result = TestResult(passed=False, summary="fail")
    assert result.passed is False
    assert result.failures == []


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == "anthropic"
    assert settings.max_cycles == 0
    assert settings.workspace is not None
