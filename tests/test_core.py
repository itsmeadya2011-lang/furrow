import pytest

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


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"
    assert Provider.GOOGLE == "google"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_validation():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=0)
    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=21)
    with pytest.raises(ValidationError):
        Settings(max_cycles=-1)
    with pytest.raises(ValidationError):
        Settings(log_level="INVALID")


def test_settings_get_api_key():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    assert s.get_api_key() == "test-key"
    
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    assert s.get_api_key() == "test-key"
    
    s = Settings(provider=Provider.GOOGLE, google_api_key="test-key")
    assert s.get_api_key() == "test-key"
    
    s = Settings(provider=Provider.OLLAMA)
    assert s.get_api_key() is None


def test_plan_with_dependencies():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", dependencies=[]),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
        ],
        rationale="test"
    )
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]