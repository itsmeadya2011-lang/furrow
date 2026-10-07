"""Comprehensive tests for the Furrow codebase."""

import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from pydantic import ValidationError

from furrow.config import (
    Provider,
    TaskModel,
    Plan,
    TestResult,
    Settings,
)
from furrow.llm import LLMClient
from furrow.core.orchestrator import Orchestrator
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent


# =============================================================================
# 1. Config / Models tests
# =============================================================================

class TestConfigModels:
    """Tests for config models: TaskModel, Plan, TestResult, Settings, Provider."""

    def test_task_model_default_status_pending(self):
        """TaskModel default status is 'pending'."""
        task = TaskModel(id="1", description="test task")
        assert task.status == "pending"

    def test_task_model_default_files_empty(self):
        """TaskModel default files list is empty."""
        task = TaskModel(id="1", description="test task")
        assert task.files == []

    def test_task_model_default_dependencies_empty(self):
        """TaskModel default dependencies list is empty."""
        task = TaskModel(id="1", description="test task")
        assert task.dependencies == []

    def test_task_model_can_transition_status_to_completed(self):
        """TaskModel can transition status to 'completed'."""
        task = TaskModel(id="1", description="test task")
        task.status = "completed"
        assert task.status == "completed"

    def test_task_model_can_transition_status_to_failed(self):
        """TaskModel can transition status to 'failed'."""
        task = TaskModel(id="1", description="test task")
        task.status = "failed"
        assert task.status == "failed"

    def test_plan_with_empty_tasks_list(self):
        """Plan accepts empty tasks list."""
        plan = Plan(tasks=[], rationale="No tasks needed")
        assert plan.tasks == []
        assert plan.rationale == "No tasks needed"

    def test_plan_with_multiple_tasks(self):
        """Plan accepts multiple tasks."""
        tasks = [
            TaskModel(id="1", description="Task 1"),
            TaskModel(id="2", description="Task 2"),
            TaskModel(id="3", description="Task 3"),
        ]
        plan = Plan(tasks=tasks, rationale="Three tasks")
        assert len(plan.tasks) == 3
        assert plan.tasks[0].id == "1"
        assert plan.tasks[1].id == "2"
        assert plan.tasks[2].id == "3"

    def test_test_result_defaults(self):
        """TestResult has correct defaults."""
        result = TestResult(passed=True, summary="All good")
        assert result.passed is True
        assert result.summary == "All good"
        assert result.failures == []

    def test_settings_defaults(self):
        """Settings has correct default values."""
        settings = Settings()
        assert settings.provider == Provider.ANTHROPIC
        assert settings.max_parallel_tasks == 5
        assert settings.max_tokens == 4096
        assert settings.max_cycles == 0
        assert settings.model == "claude-sonnet-4-20250514"
        assert settings.planner_model == "claude-3-5-haiku-20241022"
        assert settings.worker_model == "claude-3-5-sonnet-20241022"
        assert settings.tester_model == "claude-3-5-sonnet-20241022"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.log_level == "INFO"

    def test_settings_with_custom_values(self):
        """Settings accepts custom values via constructor."""
        settings = Settings(
            provider=Provider.OPENAI,
            max_parallel_tasks=10,
            max_tokens=8192,
            max_cycles=3,
            model="gpt-4",
            ollama_base_url="http://custom:11434",
        )
        assert settings.provider == Provider.OPENAI
        assert settings.max_parallel_tasks == 10
        assert settings.max_tokens == 8192
        assert settings.max_cycles == 3
        assert settings.model == "gpt-4"
        assert settings.ollama_base_url == "http://custom:11434"

    def test_provider_enum_has_anthropic(self):
        """Provider enum has ANTHROPIC value."""
        assert Provider.ANTHROPIC == "anthropic"

    def test_provider_enum_has_openai(self):
        """Provider enum has OPENAI value."""
        assert Provider.OPENAI == "openai"

    def test_provider_enum_has_ollama(self):
        """Provider enum has OLLAMA value."""
        assert Provider.OLLAMA == "ollama"


# =============================================================================
# 2. LLMClient tests (mock the provider)
# =============================================================================

class TestLLMClient:
    """Tests for LLMClient class."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            openai_api_key="test-key",
        )

    @pytest.fixture
    def client(self, settings):
        """Create LLMClient instance."""
        return LLMClient(settings=settings)

    def test_complete_raises_value_error_for_unsupported_provider(self, client):
        """complete() raises ValueError for unsupported provider."""
        client.settings.provider = "unsupported_provider"  # type: ignore
        with pytest.raises(ValueError, match="Unsupported provider"):
            asyncio.run(client.complete("test prompt"))

    @pytest.mark.asyncio
    async def test_ollama_property_creates_asyncopenai_with_correct_base_url(self, client):
        """ollama property creates AsyncOpenAI with correct base_url."""
        with patch("furrow.llm.AsyncOpenAI") as mock_async_openai:
            mock_instance = AsyncMock()
            mock_async_openai.return_value = mock_instance

            ollama_client = client.ollama

            mock_async_openai.assert_called_once_with(
                api_key="ollama",
                base_url=client.settings.ollama_base_url,
            )
            assert ollama_client == mock_instance

    @pytest.mark.asyncio
    async def test_read_file_works_correctly(self, client, tmp_path):
        """read_file reads file content correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        content = await client.read_file(test_file)
        assert content == "Hello, World!"

    @pytest.mark.asyncio
    async def test_write_file_works_correctly(self, client, tmp_path):
        """write_file writes content correctly."""
        test_file = tmp_path / "output.txt"
        await client.write_file(test_file, "Test content")

        assert test_file.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_write_file_creates_parent_directories(self, client, tmp_path):
        """write_file creates parent directories if they don't exist."""
        test_file = tmp_path / "nested" / "dir" / "output.txt"
        await client.write_file(test_file, "Nested content")

        assert test_file.read_text() == "Nested content"

    def test_list_files_returns_relative_paths(self, client, tmp_path):
        """list_files returns relative paths."""
        (tmp_path / "file1.txt").write_text("1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.txt").write_text("2")

        files = client.list_files(tmp_path)
        assert "file1.txt" in files
        assert "subdir/file2.txt" in files

    def test_list_files_returns_empty_list_for_nonexistent_directory(self, client, tmp_path):
        """list_files returns empty list for non-existent directory."""
        nonexistent = tmp_path / "nonexistent"
        files = client.list_files(nonexistent)
        assert files == []

    def test_retry_decorator_applied_to_complete_anthropic(self, client):
        """Retry decorator is applied to _complete_anthropic."""
        # Check that the method has the retry wrapper from tenacity
        method = client._complete_anthropic
        # tenacity adds a 'retry' attribute or __wrapped__
        assert hasattr(method, "retry") or hasattr(method, "__wrapped__")

    def test_retry_decorator_applied_to_complete_openai(self, client):
        """Retry decorator is applied to _complete_openai."""
        method = client._complete_openai
        assert hasattr(method, "retry") or hasattr(method, "__wrapped__")

    def test_retry_decorator_applied_to_complete_ollama(self, client):
        """Retry decorator is applied to _complete_ollama."""
        method = client._complete_ollama
        assert hasattr(method, "retry") or hasattr(method, "__wrapped__")


# =============================================================================
# 3. Orchestrator tests (mock LLMClient)
# =============================================================================

class TestOrchestrator:
    """Tests for Orchestrator class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked LLMClient."""
        client = AsyncMock(spec=LLMClient)
        client.settings = Settings(max_cycles=0, max_parallel_tasks=2)
        return client

    @pytest.fixture
    def orchestrator(self, mock_client):
        """Create Orchestrator with mocked client."""
        return Orchestrator(goal="Test goal", client=mock_client)

    def test_is_done_returns_true_when_all_tasks_completed(self, orchestrator):
        """_is_done() returns True when all tasks completed."""
        orchestrator.tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="completed"),
        ]
        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_any_task_failed(self, orchestrator):
        """_is_done() returns False when any task failed."""
        orchestrator.tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="failed"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_tasks_pending(self, orchestrator):
        """_is_done() returns False when tasks are pending."""
        orchestrator.tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="pending"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_no_tasks(self, orchestrator):
        """_is_done() returns False when no tasks (empty list)."""
        orchestrator.tasks = []
        assert orchestrator._is_done() is False

    def test_get_tasks_returns_stored_tasks(self, orchestrator):
        """_get_tasks() returns stored tasks."""
        tasks = [
            TaskModel(id="1", description="Task 1"),
            TaskModel(id="2", description="Task 2"),
        ]
        orchestrator.tasks = tasks
        assert orchestrator._get_tasks() == tasks

    @pytest.mark.asyncio
    async def test_max_cycles_enforcement_stops_after_one_cycle(self, mock_client):
        """max_cycles enforcement: with max_cycles=1, loop stops after 1 cycle."""
        settings = Settings(max_cycles=1, max_parallel_tasks=2)
        mock_client.settings = settings

        # Mock planner to return empty plan (to avoid actual execution)
        mock_planner = AsyncMock()
        mock_planner.plan.return_value = Plan(tasks=[], rationale="Done")

        orchestrator = Orchestrator(goal="Test goal", client=mock_client, settings=settings)
        orchestrator.planner = mock_planner

        await orchestrator.run()

        assert orchestrator.cycles == 1
        mock_planner.plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_message_callback_called_with_progress_messages(self, mock_client):
        """on_message callback is called with progress messages."""
        messages = []

        def on_message(msg: str):
            messages.append(msg)

        settings = Settings(max_cycles=1, max_parallel_tasks=2)
        mock_client.settings = settings

        mock_planner = AsyncMock()
        mock_planner.plan.return_value = Plan(tasks=[], rationale="Done")

        orchestrator = Orchestrator(
            goal="Test goal",
            client=mock_client,
            settings=settings,
            on_message=on_message,
        )
        orchestrator.planner = mock_planner

        await orchestrator.run()

        assert len(messages) > 0
        # Check that some expected messages were sent
        assert any("Furrow" in msg for msg in messages)
        assert any("Cycle 1" in msg for msg in messages)

    @pytest.mark.asyncio
    async def test_tasks_stored_in_self_tasks_after_cycle(self, mock_client):
        """Tasks are stored in self.tasks after _cycle()."""
        settings = Settings(max_cycles=0, max_parallel_tasks=2)
        mock_client.settings = settings

        # Create tasks that will be returned by planner
        planned_tasks = [
            TaskModel(id="1", description="Task 1"),
            TaskModel(id="2", description="Task 2"),
        ]
        mock_planner = AsyncMock()
        mock_planner.plan.return_value = Plan(tasks=planned_tasks, rationale="Two tasks")

        # Mock worker to return success
        mock_worker_run = AsyncMock(return_value="Task completed")

        orchestrator = Orchestrator(goal="Test goal", client=mock_client, settings=settings)
        orchestrator.planner = mock_planner

        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_class, \
             patch("furrow.core.orchestrator.TesterAgent") as mock_tester_class:
            mock_worker_instance = AsyncMock()
            mock_worker_instance.run = mock_worker_run
            mock_worker_class.return_value = mock_worker_instance

            mock_tester_instance = AsyncMock()
            mock_tester_instance.run.return_value = TestResult(passed=True, summary="All passed", failures=[])
            mock_tester_class.return_value = mock_tester_instance

            await orchestrator._cycle()

            assert orchestrator.tasks == planned_tasks
            assert len(orchestrator.tasks) == 2


# =============================================================================
# 4. Web server test
# =============================================================================

class TestWebServer:
    """Tests for web server integration."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_async_on_message_callback(self):
        """Orchestrator with async on_message callback properly calls/awaits it."""
        messages = []

        async def async_on_message(message: str):
            messages.append(message)

        mock_client = AsyncMock(spec=LLMClient)
        mock_client.settings = Settings(max_cycles=1, max_parallel_tasks=2)

        mock_planner = AsyncMock()
        mock_planner.plan.return_value = Plan(tasks=[], rationale="Done")

        orchestrator = Orchestrator(
            goal="Test goal",
            client=mock_client,
            on_message=async_on_message,
        )
        orchestrator.planner = mock_planner

        await orchestrator.run()

        assert len(messages) > 0
        assert any("Furrow" in msg for msg in messages)


# =============================================================================
# 5. Agent tests (mock LLMClient)
# =============================================================================

class TestPlannerAgent:
    """Tests for PlannerAgent."""

    @pytest.mark.asyncio
    async def test_plan_returns_plan_when_llm_returns_valid_json(self):
        """PlannerAgent.plan() returns a Plan when LLM returns valid JSON."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.settings = Settings(planner_model="test-model")
        mock_client.complete.return_value = json.dumps({
            "tasks": [
                {"id": "1", "description": "Task 1", "files": ["file1.py"], "dependencies": []},
                {"id": "2", "description": "Task 2", "files": ["file2.py"], "dependencies": ["1"]},
            ],
            "rationale": "Two tasks planned",
        })

        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("Test goal")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "Task 1"
        assert plan.tasks[1].id == "2"
        assert plan.rationale == "Two tasks planned"
        mock_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_plan_raises_value_error_when_llm_returns_invalid_json(self):
        """PlannerAgent.plan() raises ValueError when LLM returns invalid JSON."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.settings = Settings(planner_model="test-model")
        mock_client.complete.return_value = "This is not valid JSON"

        planner = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await planner.plan("Test goal")


class TestWorkerAgent:
    """Tests for WorkerAgent."""

    @pytest.mark.asyncio
    async def test_run_returns_llm_response(self):
        """WorkerAgent.run() returns the LLM response."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.settings = Settings(worker_model="test-model")
        mock_client.complete.return_value = "Task completed successfully"

        task = TaskModel(id="1", description="Test task", files=["test.py"])
        worker = WorkerAgent(task=task, client=mock_client)

        result = await worker.run()

        assert result == "Task completed successfully"
        mock_client.complete.assert_called_once()


class TestTesterAgent:
    """Tests for TesterAgent."""

    @pytest.mark.asyncio
    async def test_run_tests_returns_no_test_runner_found_when_no_runner_available(self):
        """TesterAgent._run_tests() returns 'No test runner found.' when no test runner available."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.settings = Settings(tester_model="test-model")

        tester = TesterAgent(client=mock_client)

        # Mock all subprocess calls to raise FileNotFoundError (command not found)
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await tester._run_tests()

        assert result == "No test runner found."

    @pytest.mark.asyncio
    async def test_run_returns_test_result_from_llm_when_tests_pass(self):
        """TesterAgent.run() returns TestResult from LLM when tests pass."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.settings = Settings(tester_model="test-model")
        mock_client.complete.return_value = json.dumps({
            "passed": True,
            "summary": "All tests passed",
            "failures": [],
        })

        tester = TesterAgent(client=mock_client)

        # Mock _run_tests to return success output
        with patch.object(tester, "_run_tests", return_value="All tests passed"):
            result = await tester.run("Test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])