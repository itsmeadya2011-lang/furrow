"""Tests for furrow.agents module (WorkerAgent, PlannerAgent, TesterAgent)."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


# --- WorkerAgent._parse_edits ---

def test_parse_edits_valid_json():
    """_parse_edits correctly parses valid JSON with edits."""
    response = json.dumps({
        "edits": [{"file": "test.py", "content": "print('hello')"}],
        "summary": "Added test file",
    })
    edits, summary = WorkerAgent._parse_edits(response)
    assert len(edits) == 1
    assert edits[0]["file"] == "test.py"
    assert edits[0]["content"] == "print('hello')"
    assert summary == "Added test file"


def test_parse_edits_empty_edits():
    """_parse_edits handles an empty edits list."""
    response = json.dumps({"edits": [], "summary": "No changes needed"})
    edits, summary = WorkerAgent._parse_edits(response)
    assert edits == []
    assert summary == "No changes needed"


def test_parse_edits_invalid_json_fallback():
    """_parse_edits falls back to raw text when JSON is invalid."""
    response = "This is not JSON"
    edits, summary = WorkerAgent._parse_edits(response)
    assert edits == []
    assert summary == "This is not JSON"


def test_parse_edits_missing_edits_key():
    """_parse_edits falls back when 'edits' key is absent (valid JSON)."""
    response = json.dumps({"summary": "done"})
    edits, summary = WorkerAgent._parse_edits(response)
    assert edits == []
    assert summary == response[:500]


def test_parse_edits_missing_summary_key():
    """_parse_edits defaults summary when 'summary' is absent."""
    response = json.dumps({"edits": [{"file": "a.py", "content": "x"}]})
    edits, summary = WorkerAgent._parse_edits(response)
    assert len(edits) == 1
    assert summary == "Completed task edits."


# --- WorkerAgent.run (mocked LLM) ---

async def test_worker_run_applies_edits(tmp_path):
    """WorkerAgent.run reads files, gets LLM response, and applies edits."""
    settings = Settings(workspace=tmp_path)
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.list_files = MagicMock(return_value=["main.py"])
    mock_client.read_file = AsyncMock(return_value="# main.py")
    mock_client.write_file = AsyncMock(return_value=None)
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "edits": [{"file": "output.py", "content": "print('done')"}],
        "summary": "Created output file",
    }))

    task = TaskModel(id="1", description="Create output.py", files=["main.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result == "Created output file"
    mock_client.complete.assert_called_once()
    mock_client.write_file.assert_awaited_once_with("output.py", "print('done')")


async def test_worker_run_no_edits():
    """WorkerAgent.run handles an LLM response with no edits."""
    settings = Settings(workspace=tmp_path)
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.list_files = MagicMock(return_value=[])
    mock_client.read_file = AsyncMock(return_value="")
    mock_client.write_file = AsyncMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "edits": [],
        "summary": "No changes needed",
    }))

    task = TaskModel(id="1", description="noop")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result == "No changes needed"
    mock_client.write_file.assert_not_awaited()


async def test_worker_run_invalid_json_fallback(tmp_path):
    """WorkerAgent.run does not crash when LLM returns non-JSON."""
    settings = Settings(workspace=tmp_path)
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.list_files = MagicMock(return_value=[])
    mock_client.read_file = AsyncMock(return_value="")
    mock_client.write_file = AsyncMock()
    mock_client.complete = AsyncMock(return_value="I can't parse this")

    task = TaskModel(id="1", description="test")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert "I can't parse this" in result
    mock_client.write_file.assert_not_awaited()


async def test_worker_run_reads_multiple_files(tmp_path):
    """WorkerAgent reads all files listed in the task."""
    settings = Settings(workspace=tmp_path)
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.list_files = MagicMock(return_value=[])
    mock_client.read_file = AsyncMock(side_effect=["content_a", "content_b"])
    mock_client.write_file = AsyncMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "edits": [{"file": "out.py", "content": "x"}],
        "summary": "done",
    }))

    task = TaskModel(id="1", description="test", files=["a.py", "b.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    await worker.run()

    assert mock_client.read_file.call_count == 2


# --- PlannerAgent ---

async def test_planner_parse_valid():
    """PlannerAgent.plan parses a valid JSON plan."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "tasks": [
            {"id": "1", "description": "task a", "files": ["a.py"], "dependencies": []},
            {"id": "2", "description": "task b", "files": ["b.py"], "dependencies": ["1"]},
        ],
        "rationale": "Split into two tasks",
    }))
    mock_client.settings = Settings()

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("Build something")

    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "Split into two tasks"


async def test_planner_invalid_json_raises():
    """PlannerAgent.plan raises ValueError on non-JSON response."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value="not json at all")
    mock_client.settings = Settings()

    planner = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Build something")


# --- TesterAgent ---

async def test_tester_parse_success():
    """TesterAgent.run returns TestResult from valid JSON."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": True,
        "summary": "All tests passed",
        "failures": [],
    }))
    mock_client.settings = Settings()

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="test output"):
        result = await tester.run("test goal", [])

    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


async def test_tester_parse_failure():
    """TesterAgent.run returns TestResult for failed tests."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": False,
        "summary": "2 tests failed",
        "failures": ["test_auth.py::test_login", "test_auth.py::test_logout"],
    }))
    mock_client.settings = Settings()

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="fail output"):
        result = await tester.run("test goal", [])

    assert result.passed is False
    assert len(result.failures) == 2
    assert result.failures[0] == "test_auth.py::test_login"


async def test_tester_json_fallback_substring():
    """TesterAgent falls back to substring detection when JSON is invalid."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value="tests passed all good")
    mock_client.settings = Settings()

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="output"):
        result = await tester.run("test goal", [])

    assert result.passed is True
    assert result.failures == []


async def test_tester_no_runner_found():
    """TesterAgent handles missing test runner gracefully."""
    mock_client = MagicMock(spec=LLMClient)
    # The LLM judges the "no runner" output as failure
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": False,
        "summary": "No test runner found",
        "failures": ["No test runner configured"],
    }))
    mock_client.settings = Settings()

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="No test runner found."):
        result = await tester.run("test goal", [])

    assert result.passed is False
    assert "No test runner" in result.summary
