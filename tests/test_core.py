import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# --- Config model tests ---


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.id == "1"
    assert t.description == "test"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_provider_enum():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_override():
    s = Settings(model="gpt-4", max_cycles=3)
    assert s.model == "gpt-4"
    assert s.max_cycles == 3


# --- LLMClient tests ---


@pytest.mark.asyncio
async def test_llm_complete_routes_to_anthropic():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=s)
    client._complete_anthropic = AsyncMock(return_value="anthropic response")
    result = await client.complete("hello", system="sys", model="claude-3")
    assert result == "anthropic response"
    client._complete_anthropic.assert_called_once_with("hello", "sys", "claude-3")


@pytest.mark.asyncio
async def test_llm_complete_routes_to_openai():
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)
    client._complete_openai = AsyncMock(return_value="openai response")
    result = await client.complete("hello", model="gpt-4")
    assert result == "openai response"
    client._complete_openai.assert_called_once_with("hello", "", "gpt-4")


@pytest.mark.asyncio
async def test_llm_complete_routes_to_ollama():
    s = Settings(provider=Provider.OLLAMA, model="llama3", ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)
    client._complete_ollama = AsyncMock(return_value="ollama response")
    result = await client.complete("hello", model="llama3")
    assert result == "ollama response"
    client._complete_ollama.assert_called_once_with("hello", "", "llama3")


@pytest.mark.asyncio
async def test_llm_retry_then_success():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=s)
    client._complete_anthropic = AsyncMock(
        side_effect=[TimeoutError("timeout"), "success"]
    )
    result = await client.complete("hello")
    assert result == "success"
    assert client._complete_anthropic.call_count == 2


@pytest.mark.asyncio
async def test_llm_retry_exhausted():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=s)
    client._complete_anthropic = AsyncMock(side_effect=TimeoutError("always fails"))
    with pytest.raises(TimeoutError, match="always fails"):
        await client.complete("hello")
    assert client._complete_anthropic.call_count == 3


# --- Orchestrator tests ---


def test_orchestrator_get_tasks_returns_accumulated():
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [TaskModel(id="1", description="t1")]
    assert len(orch._get_tasks()) == 1
    assert orch._get_tasks()[0].id == "1"


def test_orchestrator_is_done_no_tasks():
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = []
    orch.last_test_result = None
    assert orch._is_done() is True


def test_orchestrator_is_done_all_complete_passing_tests():
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [TaskModel(id="1", description="t1", status="completed")]
    orch.last_test_result = TestResult(passed=True, summary="all good")
    assert orch._is_done() is True


def test_orchestrator_is_done_failing_tests():
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [TaskModel(id="1", description="t1", status="completed")]
    orch.last_test_result = TestResult(passed=False, summary="fail", failures=["test1"])
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_max_cycles_enforced():
    s = Settings(max_cycles=1)
    client = MagicMock()
    client.settings = s
    orch = Orchestrator(goal="test", client=client)
    orch._emit = AsyncMock()
    orch._is_done = MagicMock(return_value=False)
    orch._cycle = AsyncMock()

    await orch.run()

    assert orch.cycles == 1
    orch._cycle.assert_called_once()
    orch._emit.assert_any_call(
        "Reached max_cycles (1). Stopping.", style="yellow"
    )


@pytest.mark.asyncio
async def test_orchestrator_runs_until_tests_pass():
    s = Settings(max_cycles=0)
    client = MagicMock()
    client.settings = s
    orch = Orchestrator(goal="test", client=client)
    orch._emit = AsyncMock()
    orch._cycle = AsyncMock()

    # First cycle: tests fail (not done). Second: tests pass (done).
    states = {"calls": 0}

    def mock_is_done():
        states["calls"] += 1
        return states["calls"] >= 2

    orch._is_done = mock_is_done

    await orch.run()

    assert orch.cycles == 2
    assert orch._cycle.call_count == 2


# --- TesterAgent tests ---


@pytest.mark.asyncio
async def test_tester_run_parses_passing_response():
    from furrow.agents.tester import TesterAgent

    s = Settings()
    client = MagicMock()
    client.settings = s
    tester = TesterAgent(client=client)

    tester._run_tests = AsyncMock(return_value="all tests passed")
    client.complete = AsyncMock(
        return_value=json.dumps(
            {"passed": True, "summary": "All tests passed", "failures": []}
        )
    )

    result = await tester.run("test goal", [])
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_run_parses_failing_response():
    from furrow.agents.tester import TesterAgent

    s = Settings()
    client = MagicMock()
    client.settings = s
    tester = TesterAgent(client=client)

    tester._run_tests = AsyncMock(return_value="1 failed")
    client.complete = AsyncMock(
        return_value=json.dumps(
            {"passed": False, "summary": "1 test failed", "failures": ["test_foo"]}
        )
    )

    result = await tester.run("test goal", [])
    assert result.passed is False
    assert "test_foo" in result.failures


@pytest.mark.asyncio
async def test_tester_run_fallback_on_invalid_json():
    from furrow.agents.tester import TesterAgent

    s = Settings()
    client = MagicMock()
    client.settings = s
    tester = TesterAgent(client=client)

    tester._run_tests = AsyncMock(return_value="some output")
    client.complete = AsyncMock(return_value="tests look okay to me")

    result = await tester.run("test goal", [])
    # "passed" is in the response text, so it should parse as passed
    assert result.passed is True
