"""Core tests for config models and basic data structures."""

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_with_multiple_tasks():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", files=["a.py"], dependencies=[]),
            TaskModel(id="2", description="b", files=["b.py"], dependencies=["1"]),
        ],
        rationale="test plan",
    )
    assert len(plan.tasks) == 2
    assert plan.tasks[1].dependencies == ["1"]


def test_test_result_defaults():
    t = TestResult(passed=False, summary="tests failed")
    assert t.failures == []


def test_provider_enum_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.max_retries == 3


def test_settings_ollama_default_url():
    s = Settings(provider=Provider.OLLAMA)
    assert s.ollama_base_url == "http://localhost:11434"
