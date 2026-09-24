import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    """Test TaskModel default values."""
    task = TaskModel(id="1", description="Test task")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_all_fields():
    """Test TaskModel with all fields provided."""
    task = TaskModel(
        id="1",
        description="Test task",
        files=["file1.py", "file2.py"],
        dependencies=["0"],
        status="completed",
        result="Success"
    )
    assert task.files == ["file1.py", "file2.py"]
    assert task.dependencies == ["0"]
    assert task.status == "completed"
    assert task.result == "Success"


def test_test_result_with_failures():
    """Test TestResult with failures."""
    result = TestResult(
        passed=False,
        summary="Tests failed",
        failures=["test_1 failed", "test_2 failed"]
    )
    assert result.passed is False
    assert len(result.failures) == 2


def test_settings_defaults():
    """Test Settings default values."""
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.planner_model == "claude-3-5-haiku-20241022"
    assert settings.worker_model == "claude-3-5-sonnet-20241022"
    assert settings.tester_model == "claude-3-5-sonnet-20241022"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0  # 0 = unlimited


def test_settings_provider_enum():
    """Test Provider enum values."""
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_validation():
    """Test Settings validation."""
    # Should work with valid provider
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    assert settings.provider == Provider.OPENAI
    
    # Should work with Ollama
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    assert settings.provider == Provider.OLLAMA


def test_plan_model_validation():
    """Test Plan model validation."""
    # Valid plan
    plan = Plan(tasks=[], rationale="Empty plan")
    assert plan.tasks == []
    
    # Plan with multiple tasks
    tasks = [
        TaskModel(id="1", description="Task 1"),
        TaskModel(id="2", description="Task 2"),
    ]
    plan = Plan(tasks=tasks, rationale="Two tasks")
    assert len(plan.tasks) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
