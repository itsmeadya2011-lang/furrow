import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.llm import LLMClient


# === Existing tests (keep these) ===

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# === New tests ===

class TestPlannerAgent:
    """Test the PlannerAgent's JSON parsing and plan creation."""

    @pytest.mark.asyncio
    async def test_plan_parses_valid_json(self):
        """Test that planner correctly parses valid JSON response."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.complete.return_value = json.dumps({
            "tasks": [{"id": "1", "description": "Task 1", "files": ["a.py"]}],
            "rationale": "Test plan"
        })
        mock_client.settings = MagicMock(planner_model="test-model")

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("Test goal")

        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "Task 1"
        assert plan.rationale == "Test plan"

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_json(self):
        """Test that planner raises ValueError on invalid JSON."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.complete.return_value = "not json at all"
        mock_client.settings = MagicMock(planner_model="test-model")

        planner = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("Test goal")


class TestWorkerAgent:
    """Test the WorkerAgent's file writing and response parsing."""

    @pytest.mark.asyncio
    async def test_worker_writes_files_from_json_response(self):
        """Test that worker writes files when LLM returns JSON with changes."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.complete.return_value = json.dumps({
            "changes": [
                {"path": "test_file.py", "content": "print('hello')"}
            ],
            "summary": "Created test file"
        })
        mock_client.settings = MagicMock(worker_model="test-model")
        mock_client.write_file = AsyncMock()

        task = TaskModel(id="1", description="Create test file")
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "Created test file"
        mock_client.write_file.assert_called_once_with("test_file.py", "print('hello')")

    @pytest.mark.asyncio
    async def test_worker_handles_markdown_json(self):
        """Test that worker parses JSON from markdown code blocks."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.complete.return_value = '```json\n{"changes": [], "summary": "No changes needed"}\n```'
        mock_client.settings = MagicMock(worker_model="test-model")
        mock_client.write_file = AsyncMock()

        task = TaskModel(id="1", description="Check code")
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "No changes needed"

    @pytest.mark.asyncio
    async def test_worker_fallback_to_raw_text(self):
        """Test that worker returns raw text when JSON parsing fails."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.complete.return_value = "Just a plain text summary"
        mock_client.settings = MagicMock(worker_model="test-model")
        mock_client.write_file = AsyncMock()

        task = TaskModel(id="1", description="Simple task")
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "Just a plain text summary"
        mock_client.write_file.assert_not_called()


class TestWorkerResponseParsing:
    """Test the WorkerAgent's _parse_response method directly."""

    def test_parse_raw_json(self):
        """Test parsing raw JSON without markdown."""
        task = TaskModel(id="1", description="test")
        mock_client = MagicMock(spec=LLMClient)
        worker = WorkerAgent(task=task, client=mock_client)

        result = worker._parse_response('{"changes": [], "summary": "test"}')
        assert result["summary"] == "test"

    def test_parse_markdown_json(self):
        """Test parsing JSON from markdown code block."""
        task = TaskModel(id="1", description="test")
        mock_client = MagicMock(spec=LLMClient)
        worker = WorkerAgent(task=task, client=mock_client)

        result = worker._parse_response('```json\n{"changes": [], "summary": "test"}\n```')
        assert result["summary"] == "test"

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with surrounding text."""
        task = TaskModel(id="1", description="test")
        mock_client = MagicMock(spec=LLMClient)
        worker = WorkerAgent(task=task, client=mock_client)

        result = worker._parse_response('Here is the result:\n{"changes": [], "summary": "test"}\nDone!')
        assert result["summary"] == "test"

    def test_parse_invalid_json_raises(self):
        """Test that invalid JSON raises an exception."""
        task = TaskModel(id="1", description="test")
        mock_client = MagicMock(spec=LLMClient)
        worker = WorkerAgent(task=task, client=mock_client)

        with pytest.raises(json.JSONDecodeError):
            worker._parse_response("not json at all")


class TestOrchestratorLogic:
    """Test the Orchestrator's loop logic without running full cycles."""

    def test_get_tasks_returns_empty_when_no_plan(self):
        """Test _get_tasks returns empty list when no plan is set."""
        mock_client = MagicMock(spec=LLMClient)
        orch = Orchestrator(goal="test", client=mock_client)

        assert orch._get_tasks() == []

    def test_get_tasks_returns_plan_tasks(self):
        """Test _get_tasks returns tasks from current plan."""
        from furrow.core.orchestrator import Orchestrator
        mock_client = MagicMock(spec=LLMClient)
        orch = Orchestrator(goal="test", client=mock_client)
        orch.current_plan = Plan(
            tasks=[TaskModel(id="1", description="test")],
            rationale="test"
        )

        assert len(orch._get_tasks()) == 1

    def test_is_done_returns_true_when_no_tasks(self):
        """Test _is_done returns True when there are no tasks."""
        from furrow.core.orchestrator import Orchestrator
        mock_client = MagicMock(spec=LLMClient)
        orch = Orchestrator(goal="test", client=mock_client)

        assert orch._is_done() is True

    def test_is_done_returns_true_when_tests_passed(self):
        """Test _is_done returns True when tests have passed."""
        from furrow.core.orchestrator import Orchestrator
        mock_client = MagicMock(spec=LLMClient)
        orch = Orchestrator(goal="test", client=mock_client)
        orch.current_plan = Plan(
            tasks=[TaskModel(id="1", description="test", status="completed")],
            rationale="test"
        )
        orch.test_passed = True

        assert orch._is_done() is True

    def test_is_done_returns_false_when_tests_failed(self):
        """Test _is_done returns False when tests have failed."""
        from furrow.core.orchestrator import Orchestrator
        mock_client = MagicMock(spec=LLMClient)
        orch = Orchestrator(goal="test", client=mock_client)
        orch.current_plan = Plan(
            tasks=[TaskModel(id="1", description="test", status="completed")],
            rationale="test"
        )
        orch.test_passed = False

        assert orch._is_done() is False

    def test_notify_callback(self):
        """Test that _notify calls the on_update callback."""
        from furrow.core.orchestrator import Orchestrator
        mock_client = MagicMock(spec=LLMClient)
        messages = []
        orch = Orchestrator(goal="test", client=mock_client, on_update=messages.append)

        orch._notify("test message")

        assert messages == ["test message"]

    def test_notify_no_callback(self):
        """Test that _notify does nothing when no callback is set."""
        from furrow.core.orchestrator import Orchestrator
        mock_client = MagicMock(spec=LLMClient)
        orch = Orchestrator(goal="test", client=mock_client)

        # Should not raise
        orch._notify("test message")


class TestLLMClient:
    """Test the LLMClient's file operations."""

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        """Test reading a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        client = LLMClient()
        content = await client.read_file(test_file)

        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        """Test writing a file."""
        test_file = tmp_path / "subdir" / "test.txt"

        client = LLMClient()
        await client.write_file(test_file, "hello world")

        assert test_file.read_text() == "hello world"

    def test_list_files(self, tmp_path):
        """Test listing files in a directory."""
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.txt").write_text("c")

        client = LLMClient()
        files = client.list_files(tmp_path)

        assert sorted(files) == ["a.txt", "b.txt", "sub/c.txt"]

    def test_list_files_nonexistent(self):
        """Test listing files in non-existent directory returns empty list."""
        client = LLMClient()
        files = client.list_files("/nonexistent/path")

        assert files == []


class TestSettings:
    """Test the Settings configuration."""

    def test_default_settings(self):
        """Test default settings values."""
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.log_level == "INFO"

    def test_settings_from_env(self, monkeypatch):
        """Test settings loaded from environment variables."""
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "10")
        monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")

        s = Settings()
        assert s.provider == Provider.OPENAI
        assert s.max_parallel_tasks == 10
        assert s.log_level == "DEBUG"
