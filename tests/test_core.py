import pytest
from pathlib import Path
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test task")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_files():
    task = TaskModel(id="1", description="test", files=["src/main.py", "tests/test_main.py"])
    assert len(task.files) == 2
    assert "src/main.py" in task.files


def test_settings_defaults():
    # Test with required env vars mocked
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        settings = Settings()
        assert settings.provider == Provider.ANTHROPIC
        assert settings.max_parallel_tasks == 5
        assert settings.log_level == "INFO"
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def test_settings_log_level_validation():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def test_settings_max_parallel_tasks_validation():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        with pytest.raises(ValueError):
            Settings(max_parallel_tasks=0)
        with pytest.raises(ValueError):
            Settings(max_parallel_tasks=100)
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def test_settings_ollama_url_validation():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        with pytest.raises(ValueError):
            Settings(provider=Provider.OLLAMA, ollama_base_url="invalid-url")
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def test_settings_api_key_validation():
    # Test that missing API key raises error
    import os
    # Clear any existing keys
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        if key in os.environ:
            del os.environ[key]
    
    try:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
            Settings(provider=Provider.ANTHROPIC)
    finally:
        pass
    
    try:
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            Settings(provider=Provider.OPENAI)
    finally:
        pass


def test_llm_client_initialization():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        client = LLMClient()
        assert client.settings.provider == Provider.ANTHROPIC
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def test_llm_client_list_files():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        client = LLMClient()
        files = client.list_files(".")
        assert isinstance(files, list)
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


@pytest.mark.asyncio
async def test_llm_client_read_write_file(tmp_path):
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        client = LLMClient()
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!"
        
        await client.write_file(test_file, test_content)
        content = await client.read_file(test_file)
        assert content == test_content
    finally:
        del os.environ["ANTHROPIC_API_KEY"]