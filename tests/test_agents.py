import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult


@pytest.mark.asyncio
async def test_plan_success():
    mock_client = AsyncMock()
    mock_client.settings = Settings(planner_model="test-model")
    plan_data = {
        "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
        "rationale": "ok",
    }
    mock_client.complete.return_value = json.dumps(plan_data)
    agent = PlannerAgent(client=mock_client)
    result = await agent.plan("build a house")
    assert isinstance(result, Plan)
    assert len(result.tasks) == 1
    assert result.tasks[0].description == "do thing"
    assert result.rationale == "ok"
    args, kwargs = mock_client.complete.call_args
    assert kwargs["model"] == "test-model"


@pytest.mark.asyncio
async def test_plan_invalid_json():
    mock_client = AsyncMock()
    mock_client.settings = Settings(planner_model="test-model")
    mock_client.complete.return_value = "not json"
    agent = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        await agent.plan("build a house")


@pytest.mark.asyncio
async def test_plan_empty_tasks():
    mock_client = AsyncMock()
    mock_client.settings = Settings(planner_model="test-model")
    plan_data = {"tasks": [], "rationale": "nothing to do"}
    mock_client.complete.return_value = json.dumps(plan_data)
    agent = PlannerAgent(client=mock_client)
    result = await agent.plan("build a house")
    assert isinstance(result, Plan)
    assert len(result.tasks) == 0
    assert result.rationale == "nothing to do"


@pytest.mark.asyncio
async def test_worker_returns_result():
    task = TaskModel(id="1", description="do thing")
    mock_client = AsyncMock()
    mock_client.settings = Settings(worker_model="test-model")
    mock_client.complete.return_value = "implementation done"
    agent = WorkerAgent(task=task, client=mock_client)
    result = await agent.run()
    assert result == "implementation done"
    args, kwargs = mock_client.complete.call_args
    assert kwargs["model"] == "test-model"


@pytest.mark.asyncio
async def test_worker_prompt_includes_task():
    task = TaskModel(id="1", description="write tests", files=["tests/test_x.py"])
    mock_client = AsyncMock()
    mock_client.settings = Settings(worker_model="test-model")
    mock_client.complete.return_value = "done"
    agent = WorkerAgent(task=task, client=mock_client)
    await agent.run()
    prompt = mock_client.complete.call_args.args[0]
    assert "write tests" in prompt
    assert "tests/test_x.py" in prompt
    assert "Files to touch:" in prompt


@pytest.mark.asyncio
async def test_tester_success():
    mock_client = AsyncMock()
    mock_client.settings = Settings(tester_model="test-model")
    test_output = "all passed"
    result_data = {"passed": True, "summary": "ok", "failures": []}
    mock_client.complete.return_value = json.dumps(result_data)
    agent = TesterAgent(client=mock_client)
    agent._run_tests = AsyncMock(return_value=test_output)
    result = await agent.run("goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "ok"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_json_parse():
    mock_client = AsyncMock()
    mock_client.settings = Settings(tester_model="test-model")
    test_output = "some output"
    result_data = {"passed": True, "summary": "all good", "failures": ["x"]}
    mock_client.complete.return_value = json.dumps(result_data)
    agent = TesterAgent(client=mock_client)
    agent._run_tests = AsyncMock(return_value=test_output)
    result = await agent.run("goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == ["x"]


@pytest.mark.asyncio
async def test_tester_fallback_on_invalid_json():
    mock_client = AsyncMock()
    mock_client.settings = Settings(tester_model="test-model")
    mock_client.complete.return_value = "no json here"
    agent = TesterAgent(client=mock_client)
    agent._run_tests = AsyncMock(return_value="test output")
    result = await agent.run("goal", [])
    assert result.passed is False
    assert result.summary == "no json here"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_test_exception():
    mock_client = AsyncMock()
    mock_client.settings = Settings(tester_model="test-model")
    agent = TesterAgent(client=mock_client)
    agent._run_tests = AsyncMock(side_effect=RuntimeError("test failed"))
    result = await agent.run("goal", [])
    assert result.passed is False
    assert result.summary == "test failed"
    assert result.failures == ["test failed"]


@pytest.mark.asyncio
async def test_run_tests_finds_pytest():
    agent = TesterAgent()
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"passed\n", b"")

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess, \
         patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
        mock_subprocess.return_value = mock_proc
        mock_wait.return_value = (b"passed\n", b"")
        result = await agent._run_tests()

    assert result == "passed\n"
    mock_subprocess.assert_called_once_with(
        "pytest", "-q",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


@pytest.mark.asyncio
async def test_run_tests_no_runner():
    agent = TesterAgent()

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=FileNotFoundError()):
        result = await agent._run_tests()

    assert result == "No test runner found."


@pytest.mark.asyncio
async def test_run_tests_timeout():
    agent = TesterAgent()
    mock_proc = AsyncMock()
    mock_proc.kill = MagicMock()

    async def _success_wait_for(*args, **kwargs):
        return (b"passed\n", b"")

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess, \
         patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
        mock_subprocess.return_value = mock_proc
        mock_wait.side_effect = [asyncio.TimeoutError(), _success_wait_for]
        result = await agent._run_tests()

    assert result == "passed\n"
    assert mock_subprocess.call_count == 2
    mock_proc.kill.assert_called_once()
