import httpx
import pytest
import tenacity
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.agents.tester import TesterAgent
from furrow.config import Plan, TaskModel, TestResult, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient, _is_retryable_http_error


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_topological_sort_no_deps():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
        TaskModel(id="3", description="task 3"),
    ]
    waves = orch._topological_sort(tasks)
    assert len(waves) == 1
    assert len(waves[0]) == 3
    assert {t.id for t in waves[0]} == {"1", "2", "3"}


def test_topological_sort_linear():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
        TaskModel(id="3", description="task 3", dependencies=["2"]),
    ]
    waves = orch._topological_sort(tasks)
    assert len(waves) == 3
    assert [t.id for t in waves[0]] == ["1"]
    assert [t.id for t in waves[1]] == ["2"]
    assert [t.id for t in waves[2]] == ["3"]


def test_topological_sort_diamond():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
        TaskModel(id="3", description="task 3", dependencies=["1", "2"]),
        TaskModel(id="4", description="task 4", dependencies=["3"]),
    ]
    waves = orch._topological_sort(tasks)
    assert len(waves) == 3
    assert [t.id for t in waves[0]] == ["1", "2"]
    assert [t.id for t in waves[1]] == ["3"]
    assert [t.id for t in waves[2]] == ["4"]


def test_topological_sort_circular():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="task 1", dependencies=["2"]),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
    ]
    waves = orch._topological_sort(tasks)
    all_ids = {t.id for wave in waves for t in wave}
    assert all_ids == {"1", "2"}
    assert len(waves) == 2


def test_is_done_no_tasks():
    orch = Orchestrator(goal="test")
    assert orch._is_done() is True


def test_is_done_all_completed_passed():
    orch = Orchestrator(goal="test")
    orch.current_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    orch.last_test_passed = True
    assert orch._is_done() is True


def test_is_done_any_failed():
    orch = Orchestrator(goal="test")
    orch.current_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    orch.last_test_passed = True
    assert orch._is_done() is False


def test_is_done_tests_failed():
    orch = Orchestrator(goal="test")
    orch.current_tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    orch.last_test_passed = False
    assert orch._is_done() is False


def test_is_retryable_http_error_5xx():
    response = MagicMock()
    response.status_code = 500
    exc = httpx.HTTPStatusError("server error", request=MagicMock(), response=response)
    assert _is_retryable_http_error(exc) is True


def test_is_retryable_http_error_4xx():
    response = MagicMock()
    response.status_code = 404
    exc = httpx.HTTPStatusError("not found", request=MagicMock(), response=response)
    assert _is_retryable_http_error(exc) is False


async def test_llm_retry_succeeds_after_failures():
    client = LLMClient(settings=Settings())
    mock_func = AsyncMock(
        side_effect=[
            httpx.TimeoutException("timeout"),
            httpx.TimeoutException("timeout"),
            "ok",
        ]
    )
    with patch("furrow.llm.wait_exponential", return_value=tenacity.wait_fixed(0)):
        result = await client._run_with_retry(mock_func)
    assert result == "ok"
    assert mock_func.call_count == 3


async def test_tester_includes_tasks_in_prompt():
    tasks = [
        TaskModel(id="1", description="First task"),
        TaskModel(id="2", description="Second task"),
    ]
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
    tester = TesterAgent(client=mock_client)
    with patch.object(TesterAgent, "_run_tests", new=AsyncMock(return_value="test output")):
        await tester.run("test goal", tasks)
    call_args = mock_client.complete.call_args
    prompt = call_args.args[0]
    assert "Task 1: First task" in prompt
    assert "Task 2: Second task" in prompt
