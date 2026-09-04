from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


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


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_defaults():
    settings = Settings()
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.provider == Provider.ANTHROPIC
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.log_level == "INFO"
