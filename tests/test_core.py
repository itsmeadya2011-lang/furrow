import pytest
from unittest.mock import AsyncMock, patch

from furrow.config import Plan, TaskModel, TestResult, Settings
from furrow.llm import LLMClient


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


def test_settings_defaults():
    s = Settings()
    assert s.provider.value == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.llm_timeout == 120
    assert s.llm_max_tokens == 4096


@pytest.mark.asyncio
async def test_llm_client_unsupported_provider():
    s = Settings(provider=Settings(provider="anthropic").provider)
    client = LLMClient(settings=s)
    client.settings.provider = "invalid"
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test")


@pytest.mark.asyncio
async def test_llm_client_missing_anthropic_key():
    s = Settings(provider="anthropic", anthropic_api_key=None)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic


@pytest.mark.asyncio
async def test_llm_client_list_files_empty():
    client = LLMClient()
    result = client.list_files("/nonexistent/path")
    assert result == []


@pytest.mark.asyncio
async def test_llm_client_list_files(tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    client = LLMClient()
    result = client.list_files(tmp_path)
    assert "test.py" in result


@pytest.mark.asyncio
async def test_worker_reads_context_files(tmp_path):
    test_file = tmp_path / "auth.py"
    test_file.write_text("def login(): pass")
    task = TaskModel(id="1", description="Add JWT", files=["auth.py"])
    client = LLMClient()
    # Verify the client can read files that the worker will use
    content = await client.read_file(test_file)
    assert "login" in content
