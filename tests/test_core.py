"""Comprehensive tests for Furrow components."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestProvider:
    def test_provider_enum_values(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"

    def test_provider_is_str_enum(self):
        assert isinstance(Provider.ANTHROPIC, str)
        assert Provider.OLLAMA.value == "ollama"


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_cycles == 0  # 0 = unlimited by default
        assert s.max_parallel_tasks == 5
        assert s.ollama_base_url == "http://localhost:11434"

    def test_custom_settings(self):
        s = Settings(provider=Provider.OPENAI, openai_api_key="test-key", max_cycles=3)
        assert s.provider == Provider.OPENAI
        assert s.openai_api_key == "test-key"
        assert s.max_cycles == 3


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestTaskModel:
    def test_default_status(self):
        t = TaskModel(id="1", description="test task")
        assert t.status == "pending"
        assert t.result is None
        assert t.files == []
        assert t.dependencies == []

    def test_completed_status(self):
        t = TaskModel(id="1", description="test task", status="completed", result="done")
        assert t.status == "completed"
        assert t.result == "done"


class TestPlan:
    def test_plan_with_tasks(self):
        tasks = [TaskModel(id="1", description="task 1"), TaskModel(id="2", description="task 2")]
        plan = Plan(tasks=tasks, rationale="Test plan")
        assert len(plan.tasks) == 2
        assert plan.rationale == "Test plan"

    def test_plan_empty_tasks(self):
        plan = Plan(tasks=[], rationale="Nothing to do")
        assert len(plan.tasks) == 0


# ---------------------------------------------------------------------------
# Orchestrator tests (pure logic, no LLM required)
# ---------------------------------------------------------------------------


class TestOrchestratorState:
    def test_initial_state(self):
        orch = Orchestrator(goal="do something")
        assert orch.goal == "do something"
        assert orch.cycles == 0
        assert orch._tasks == []
        assert orch._tests_passed is False
        assert orch._stop_requested is False

    def test_get_tasks_returns_empty_initially(self):
        orch = Orchestrator(goal="do something")
        assert orch._get_tasks() == []

    def test_max_cycles_unlimited_by_default(self):
        orch = Orchestrator(goal="do something")
        # max_cycles defaults to 0 (unlimited)
        assert orch.settings.max_cycles == 0


class TestOrchestratorIsDone:
    """Test the _is_done() logic which was the critical bug."""

    def test_is_done_no_tasks(self):
        """With no tasks, should be considered done."""
        orch = Orchestrator(goal="done")
        assert orch._is_done() is True

    def test_is_done_all_completed_tests_passed(self):
        """All tasks completed and tests passed => done."""
        orch = Orchestrator(goal="test")
        orch._tasks = [
            TaskModel(id="1", description="task 1", status="completed", result="ok"),
            TaskModel(id="2", description="task 2", status="completed", result="ok"),
        ]
        orch._tests_passed = True
        assert orch._is_done() is True

    def test_is_done_all_completed_no_tests(self):
        """All tasks completed but tests not passed => not done (should retry)."""
        orch = Orchestrator(goal="test")
        orch._tasks = [
            TaskModel(id="1", description="task 1", status="completed", result="ok"),
        ]
        orch._tests_passed = False
        assert orch._is_done() is False

    def test_is_done_with_pending_tasks(self):
        """Tasks still pending => not done."""
        orch = Orchestrator(goal="test")
        orch._tasks = [
            TaskModel(id="1", description="task 1", status="pending"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        orch._tests_passed = True
        assert orch._is_done() is False

    def test_is_done_with_failed_tasks(self):
        """Any failed tasks => not done (should retry)."""
        orch = Orchestrator(goal="test")
        orch._tasks = [
            TaskModel(id="1", description="task 1", status="failed", result="error"),
        ]
        orch._tests_passed = False
        assert orch._is_done() is False

    def test_is_done_all_pending(self):
        """All tasks pending => not done."""
        orch = Orchestrator(goal="test")
        orch._tasks = [
            TaskModel(id="1", description="task 1", status="pending"),
        ]
        assert orch._is_done() is False


class TestOrchestratorMaxCycles:
    @pytest.mark.asyncio
    @patch("furrow.core.orchestrator.Orchestrator._cycle")
    @patch("furrow.core.orchestrator.Orchestrator._is_done")
    async def test_max_cycles_terminates(self, mock_is_done, mock_cycle):
        """Orchestrator should stop after max_cycles even if not done."""
        mock_is_done.return_value = False
        mock_cycle.return_value = None

        settings_obj = Settings(max_cycles=2)
        orch = Orchestrator(goal="test goal", settings_obj=settings_obj)
        await orch.run()

        assert mock_cycle.call_count == 2
        assert orch.cycles == 2


class TestOrchestratorStopRequest:
    @pytest.mark.asyncio
    @patch("furrow.core.orchestrator.Orchestrator._cycle")
    @patch("furrow.core.orchestrator.Orchestrator._is_done")
    async def test_stop_request_terminates(self, mock_is_done, mock_cycle):
        """Orchestrator should stop when stop is requested."""
        mock_is_done.return_value = False
        mock_cycle.return_value = None

        orch = Orchestrator(goal="test goal")

        # Request stop after the first cycle
        call_count = 0

        async def stop_after_first_cycle():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                orch.request_stop()

        orch._cycle = stop_after_first_cycle
        await orch.run()

        assert orch._stop_requested is True
        assert orch.cycles >= 1


class TestOrchestratorCycleError:
    @pytest.mark.asyncio
    async def test_cycle_error_does_not_crash(self):
        """If _cycle raises an exception, the loop should continue."""
        orch = Orchestrator(goal="test")
        # Pre-populate a pending task so _is_done() returns False after cycle errors
        orch._tasks = [TaskModel(id="1", description="task", status="pending")]

        call_count = 0

        async def failing_cycle():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Simulated cycle error")
            # On third call, set tasks to completed and tests passed to stop
            orch._tasks = [TaskModel(id="1", description="task", status="completed")]
            orch._tests_passed = True

        orch._cycle = failing_cycle

        # Should survive errors and eventually stop
        await orch.run()
        assert call_count == 3


class TestOrchestratorCallback:
    @pytest.mark.asyncio
    @patch("furrow.core.orchestrator.Orchestrator._cycle")
    @patch("furrow.core.orchestrator.Orchestrator._is_done")
    async def test_output_callback_invoked(self, mock_is_done, mock_cycle):
        """Output callback should be invoked when set."""
        mock_is_done.return_value = True
        mock_cycle.return_value = None

        orch = Orchestrator(goal="test")
        received = []

        def callback(msg: str):
            received.append(msg)

        orch.set_output_callback(callback)
        await orch.run()

        assert len(received) > 0
        assert any("Furrow" in msg for msg in received)


# ---------------------------------------------------------------------------
# WorkerAgent tests
# ---------------------------------------------------------------------------


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_parses_json_response(self):
        """Worker should parse JSON response and write files."""
        from furrow.agents.worker import WorkerAgent

        mock_client = MagicMock()
        mock_client.settings.worker_model = "test-model"
        mock_client.settings.provider = Provider.ANTHROPIC

        response_data = {
            "changes": [
                {"path": "/tmp/test_worker_output.py", "content": "print('hello')"},
            ],
            "summary": "Created test file",
        }
        mock_client.complete = AsyncMock(return_value=json.dumps(response_data))
        mock_client.read_file = AsyncMock(side_effect=FileNotFoundError())
        mock_client.write_file = AsyncMock()

        task = TaskModel(id="1", description="Create a test file", files=["/tmp/test_worker_output.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "Created test file"
        mock_client.write_file.assert_called_once_with("/tmp/test_worker_output.py", "print('hello')")

    @pytest.mark.asyncio
    async def test_worker_strips_markdown_fences(self):
        """Worker should handle markdown code fence wrapping."""
        from furrow.agents.worker import WorkerAgent

        mock_client = MagicMock()
        mock_client.settings.worker_model = "test-model"

        response_data = {
            "changes": [],
            "summary": "No changes needed",
        }
        # LLM wraps JSON in code fences
        raw_response = f"```json\n{json.dumps(response_data)}\n```"
        mock_client.complete = AsyncMock(return_value=raw_response)
        mock_client.read_file = AsyncMock(side_effect=FileNotFoundError())
        mock_client.write_file = AsyncMock()

        task = TaskModel(id="1", description="No-op task")
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "No changes needed"

    @pytest.mark.asyncio
    async def test_worker_handles_json_parse_failure(self):
        """Worker should return a graceful message on JSON parse failure."""
        from furrow.agents.worker import WorkerAgent

        mock_client = MagicMock()
        mock_client.settings.worker_model = "test-model"
        mock_client.complete = AsyncMock(return_value="This is not JSON at all")
        mock_client.read_file = AsyncMock(side_effect=FileNotFoundError())
        mock_client.write_file = AsyncMock()

        task = TaskModel(id="1", description="Bad response task")
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert "could not parse" in result.lower()
        mock_client.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_reads_existing_files(self):
        """Worker should read existing file contents to provide context."""
        from furrow.agents.worker import WorkerAgent

        mock_client = MagicMock()
        mock_client.settings.worker_model = "test-model"
        mock_client.complete = AsyncMock(
            return_value=json.dumps({"changes": [], "summary": "ok"})
        )
        mock_client.read_file = AsyncMock(return_value="# existing code")
        mock_client.write_file = AsyncMock()

        task = TaskModel(id="1", description="Modify file", files=["existing.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        mock_client.read_file.assert_called_once_with("existing.py")


# ---------------------------------------------------------------------------
# TesterAgent tests
# ---------------------------------------------------------------------------


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_no_test_runner_is_pass(self):
        """When no test runner is found, tests should pass."""
        from furrow.agents.tester import TesterAgent

        mock_client = MagicMock()
        mock_client.settings.tester_model = "test-model"
        mock_client.settings.workspace = Path.cwd()
        mock_client.complete = AsyncMock(
            return_value=json.dumps({"passed": True, "summary": "ok", "failures": []})
        )

        tester = TesterAgent(client=mock_client)
        tester._run_tests = AsyncMock(return_value="No test runner found.")

        result = await tester.run("test goal", [])

        assert result.passed is True
        assert "No test runner" in result.summary

    @pytest.mark.asyncio
    async def test_tester_parses_json_response(self):
        """Tester should parse JSON response from LLM."""
        from furrow.agents.tester import TesterAgent

        mock_client = MagicMock()
        mock_client.settings.tester_model = "test-model"
        mock_client.settings.workspace = Path.cwd()

        mock_client.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "passed": False,
                    "summary": "2 tests failed",
                    "failures": ["test_foo failed", "test_bar failed"],
                }
            )
        )

        tester = TesterAgent(client=mock_client)
        tester._run_tests = AsyncMock(return_value="test output")

        result = await tester.run("test goal", [])

        assert result.passed is False
        assert len(result.failures) == 2

    @pytest.mark.asyncio
    async def test_tester_fallback_on_bad_json(self):
        """Tester should fall back to heuristic parsing when JSON is invalid."""
        from furrow.agents.tester import TesterAgent

        mock_client = MagicMock()
        mock_client.settings.tester_model = "test-model"
        mock_client.settings.workspace = Path.cwd()
        mock_client.complete = AsyncMock(return_value="All tests passed! No failures.")

        tester = TesterAgent(client=mock_client)
        tester._run_tests = AsyncMock(return_value="test output")

        result = await tester.run("test goal", [])

        assert result.passed is True
        assert "All tests" in result.summary

    @pytest.mark.asyncio
    async def test_tester_strips_markdown_fences(self):
        """Tester should strip markdown code fences from LLM response."""
        from furrow.agents.tester import TesterAgent

        mock_client = MagicMock()
        mock_client.settings.tester_model = "test-model"
        mock_client.settings.workspace = Path.cwd()

        response_data = {"passed": True, "summary": "ok", "failures": []}
        raw_response = f"```json\n{json.dumps(response_data)}\n```"
        mock_client.complete = AsyncMock(return_value=raw_response)

        tester = TesterAgent(client=mock_client)
        tester._run_tests = AsyncMock(return_value="test output")

        result = await tester.run("test goal", [])

        assert result.passed is True


# ---------------------------------------------------------------------------
# LLMClient tests
# ---------------------------------------------------------------------------


class TestLLMClient:
    def test_unsupported_provider_raises(self):
        """LLM client with unknown provider should raise ValueError."""
        from furrow.llm import LLMClient

        class FakeProvider:
            pass

        # Create a settings-like object with an invalid provider
        mock_settings = MagicMock()
        mock_settings.provider = FakeProvider()
        mock_settings.model = "test"
        client = LLMClient.__new__(LLMClient)
        client.settings = mock_settings

        import asyncio

        async def run_test():
            with pytest.raises(ValueError, match="Unsupported provider"):
                await client.complete("test prompt")

        asyncio.run(run_test())

    @pytest.mark.asyncio
    async def test_ollama_provider_routing(self):
        """Ollama provider should be routed to _complete_ollama."""
        from furrow.llm import LLMClient

        mock_settings = MagicMock()
        mock_settings.provider = Provider.OLLAMA
        mock_settings.model = "test-ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"

        client = LLMClient.__new__(LLMClient)
        client.settings = mock_settings
        client._httpx = MagicMock()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Ollama response"}
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._httpx = mock_http

        result = await client.complete("test prompt", system="test system")

        assert result == "Ollama response"
        mock_http.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_ollama_malformed_response(self):
        """Ollama should handle malformed response gracefully."""
        from furrow.llm import LLMClient

        mock_settings = MagicMock()
        mock_settings.provider = Provider.OLLAMA
        mock_settings.model = "test-ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"

        client = LLMClient.__new__(LLMClient)
        client.settings = mock_settings

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._httpx = mock_http

        result = await client.complete("test prompt")
        assert result == ""


# ---------------------------------------------------------------------------
# Integration: Orchestrator with mocked agents
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    @pytest.mark.asyncio
    async def test_full_cycle_with_mocked_planner(self):
        """Test a full orchestrator cycle with mocked planner, worker, and tester."""
        settings_obj = Settings(max_cycles=1)
        orch = Orchestrator(goal="Add a hello function", settings_obj=settings_obj)

        # Mock the planner to return a plan
        plan = Plan(
            tasks=[TaskModel(id="1", description="Create hello.py", files=["hello.py"])],
            rationale="Simple task",
        )
        orch.planner.plan = AsyncMock(return_value=plan)

        # Mock the worker to return a summary
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value="Created hello.py")
            MockWorker.return_value = mock_instance

            # Mock the tester to return passed
            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                mock_tester = MagicMock()
                mock_tester.run = AsyncMock(
                    return_value=TestResult(passed=True, summary="All passed", failures=[])
                )
                MockTester.return_value = mock_tester

                await orch._cycle()

        # Verify planner was called
        orch.planner.plan.assert_called_once()

        # Verify tasks were tracked
        assert len(orch._tasks) == 1
        assert orch._tasks[0].status == "completed"
        assert orch._tasks[0].result == "Created hello.py"

        # Verify tests passed
        assert orch._tests_passed is True

    @pytest.mark.asyncio
    async def test_cycle_with_failed_tests_continues(self):
        """When tests fail, the goal should be updated and loop should not stop."""
        orch = Orchestrator(goal="Fix the bug")
        orch._tasks = []  # Ensure clean state

        plan = Plan(
            tasks=[TaskModel(id="1", description="Fix bug", files=["bug.py"])],
            rationale="Fix it",
        )
        orch.planner.plan = AsyncMock(return_value=plan)

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value="Fixed bug")
            MockWorker.return_value = mock_instance

            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                mock_tester = MagicMock()
                mock_tester.run = AsyncMock(
                    return_value=TestResult(
                        passed=False,
                        summary="Tests failed",
                        failures=["test_foo failed"],
                    )
                )
                MockTester.return_value = mock_tester

                await orch._cycle()

        # Tests should not have passed
        assert orch._tests_passed is False
        # Goal should be updated with fix instructions
        assert "Fix failing tests" in orch.goal
