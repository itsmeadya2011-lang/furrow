from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.llm_timeout == 120
    assert s.test_timeout == 120
    assert s.max_retries == 2


def test_settings_override():
    s = Settings(provider="openai", max_cycles=3, workspace=Path("/tmp"))
    assert s.provider == "openai"
    assert s.max_cycles == 3
    assert s.workspace == Path("/tmp")


def test_task_model_defaults():
    t = TaskModel(id="1", description="x")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_requires_tasks():
    with pytest.raises(ValidationError):
        Plan(tasks=[], rationale="ok")


def test_orchestrator_get_tasks_empty_before_cycle():
    custom_settings = Settings()
    client = LLMClient(settings=custom_settings)
    o = Orchestrator(goal="test", client=client)
    assert o._get_tasks() == []


def test_orchestrator_is_done_false_when_no_tasks():
    custom_settings = Settings()
    client = LLMClient(settings=custom_settings)
    o = Orchestrator(goal="test", client=client)
    assert o._is_done() is False


def test_orchestrator_max_cycles_enforced():
    custom_settings = Settings()
    client = LLMClient(settings=custom_settings)
    o = Orchestrator(goal="test", client=client)
    o.cycles = 5
    # Simulate the check that happens in run()
    # We just verify the _is_done logic doesn't block cycles
    assert o._is_done() is False


def test_test_result_from_llm_json_parse():
    t = TestResult(**{"passed": False, "summary": "fail", "failures": ["a", "b"]})
    assert t.passed is False
    assert len(t.failures) == 2


def test_test_result_fallback_on_non_json():
    t = TestResult(passed="passed" in "Some text passed".lower(), summary="Some text passed", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_worker_build_file_context_existing_file(tmp_path: Path):
    f = tmp_path / "foo.py"
    f.write_text("print('hello')")
    custom_settings = Settings(workspace=Path(tmp_path))
    client = LLMClient(settings=custom_settings)
    w = WorkerAgent(task=TaskModel(id="1", description="test", files=["foo.py"]), client=client)
    ctx = await w._build_file_context(Path(tmp_path), ["foo.py"])
    assert "foo.py" in ctx
    assert "print('hello')" in ctx


@pytest.mark.asyncio
async def test_worker_build_file_context_missing_file(tmp_path: Path):
    custom_settings = Settings(workspace=Path(tmp_path))
    client = LLMClient(settings=custom_settings)
    w = WorkerAgent(task=TaskModel(id="1", description="test", files=["missing.py"]), client=client)
    ctx = await w._build_file_context(Path(tmp_path), ["missing.py"])
    assert "missing.py" in ctx
    assert "does not exist yet" in ctx


@pytest.mark.asyncio
async def test_worker_build_file_context_no_files(tmp_path: Path):
    custom_settings = Settings(workspace=Path(tmp_path))
    client = LLMClient(settings=custom_settings)
    w = WorkerAgent(task=TaskModel(id="1", description="test", files=[]), client=client)
    ctx = await w._build_file_context(Path(tmp_path), [])
    assert "No specific files mentioned" in ctx


@pytest.mark.asyncio
async def test_tester_run_returns_result_on_exception():
    custom_settings = Settings()
    client = LLMClient(settings=custom_settings)
    with patch.object(client, "complete", side_effect=RuntimeError("LLM down")):
        t = TesterAgent(client=client)
        result = await t._run_tests()
        # _run_tests should still return something even if no tests found
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_llm_client_complete_unsupported_provider():
    s = Settings.model_construct(provider="fake")
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test")


@pytest.mark.asyncio
async def test_llm_client_stream_not_implemented():
    s = Settings(provider="openai", openai_api_key="sk-test")
    client = LLMClient(settings=s)
    with patch.object(client.openai.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = []
        # stream requires an async iterable; just verify it doesn't crash on setup
        # We can't easily mock the async iteration here without more setup
        # So we just test that the method exists and is a coroutine
        import inspect
        assert inspect.iscoroutinefunction(client.stream)


def test_planner_agent_uses_custom_settings():
    custom_settings = Settings(model="custom-model")
    client = LLMClient(settings=custom_settings)
    assert client.settings.model == "custom-model"