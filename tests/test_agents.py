from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.prompts import TESTER_PROMPT, WORKER_PROMPT
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------


def _make_worker(task: TaskModel | None = None, client: MagicMock | None = None) -> WorkerAgent:
    task = task or TaskModel(id="1", description="implement thing")
    if client is None:
        client = MagicMock()
        client.settings.worker_model = "worker-model-x"
    return WorkerAgent(task=task, client=client)


def test_worker_calls_complete_with_tools():
    client = MagicMock()
    client.settings.worker_model = "claude-worker"
    client.complete_with_tools = AsyncMock(return_value="done")

    task = TaskModel(id="1", description="do thing")
    agent = WorkerAgent(task=task, client=client)

    result = asyncio.run(agent.run())

    assert result == "done"
    client.complete_with_tools.assert_awaited_once()
    _, kwargs = client.complete_with_tools.call_args
    assert kwargs["system"] == WORKER_PROMPT
    assert kwargs["model"] == "claude-worker"
    assert kwargs["max_iterations"] == 15


def test_worker_prompt_includes_task():
    client = MagicMock()
    client.settings.worker_model = "m"
    client.complete_with_tools = AsyncMock(return_value="ok")

    task = TaskModel(id="42", description="implement the widget")
    agent = WorkerAgent(task=task, client=client)

    asyncio.run(agent.run())

    args, _ = client.complete_with_tools.call_args
    prompt = args[0]
    assert "42" in prompt
    assert "implement the widget" in prompt


# ---------------------------------------------------------------------------
# TesterAgent
# ---------------------------------------------------------------------------


def _make_tester(client: MagicMock | None = None) -> TesterAgent:
    if client is None:
        client = MagicMock()
        client.settings.tester_model = "tester-model-y"
    return TesterAgent(client=client)


def test_tester_run_tests_no_runner():
    client = MagicMock()
    client.settings.tester_model = "m"
    tester = TesterAgent(client=client)

    # Make every candidate subprocess exec fail
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("nope")):
        result = asyncio.run(tester._run_tests())

    assert result == "No test runner found."


def test_tester_parse_valid_json():
    client = MagicMock()
    client.settings.tester_model = "m"
    client.complete = AsyncMock(
        return_value=json.dumps({"passed": True, "summary": "all good", "failures": []})
    )
    tester = TesterAgent(client=client)

    # Avoid actually running subprocesses
    with patch.object(tester, "_run_tests", AsyncMock(return_value="test output")):
        result = asyncio.run(tester.run("my goal", []))

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "all good"


def test_tester_parse_invalid_json():
    client = MagicMock()
    client.settings.tester_model = "m"
    client.complete = AsyncMock(return_value="This is not JSON at all")
    tester = TesterAgent(client=client)

    with patch.object(tester, "_run_tests", AsyncMock(return_value="test output")):
        result = asyncio.run(tester.run("my goal", []))

    assert isinstance(result, TestResult)
    # Fallback: passed based on presence of "passed" keyword
    assert result.passed is False
    assert result.summary == "This is not JSON at all"


def test_tester_uses_worker_model():
    # Per the spec: TESTER uses tester_model
    client = MagicMock()
    client.settings.tester_model = "my-tester-model"
    client.complete = AsyncMock(return_value=json.dumps({"passed": True, "summary": "ok", "failures": []}))
    tester = TesterAgent(client=client)

    with patch.object(tester, "_run_tests", AsyncMock(return_value="")):
        asyncio.run(tester.run("goal", []))

    args, kwargs = client.complete.call_args
    assert kwargs["model"] == "my-tester-model"