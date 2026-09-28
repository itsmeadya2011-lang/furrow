"""Tests for improvements to the Furrow orchestrator, config, and agents."""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigImprovements:
    """Verify new config fields are present with correct defaults."""

    def test_ollama_model_default(self):
        settings = Settings()
        assert settings.ollama_model == "llama3.1"

    def test_state_file_default(self):
        settings = Settings()
        assert isinstance(settings.state_file, Path)
        assert settings.state_file.name == ".furrow_state.json"

    def test_log_file_default(self):
        settings = Settings()
        assert isinstance(settings.log_file, Path)
        assert settings.log_file.name == "furrow.log"

    def test_provider_enum_has_ollama(self):
        assert Provider.OLLAMA == "ollama"


# ---------------------------------------------------------------------------
# Orchestrator tests — _get_tasks / task tracking
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """Return an LLMClient with a stubbed Settings (no real API keys)."""
    mock = MagicMock(spec=LLMClient)
    mock.settings = Settings()
    return mock


@pytest.fixture
def orchestrator(mock_client):
    return Orchestrator(goal="Test goal", client=mock_client)


class TestOrchestratorTaskTracking:
    """The _get_tasks() bug fix: tasks must be tracked, not lost."""

    def test_get_tasks_starts_empty(self, orchestrator):
        assert orchestrator._get_tasks() == []

    def test_get_tasks_returns_stored_tasks(self, orchestrator):
        tasks = [
            TaskModel(id="1", description="task a"),
            TaskModel(id="2", description="task b"),
        ]
        orchestrator._tasks = tasks
        assert len(orchestrator._get_tasks()) == 2
        assert orchestrator._get_tasks()[0].id == "1"


class TestOrchestratorIsDone:
    """Verify the fixed _is_done() termination logic."""

    def test_done_when_no_tasks(self, orchestrator):
        orchestrator._tasks = []
        assert orchestrator._is_done() is True

    def test_not_done_when_tasks_pending(self, orchestrator):
        orchestrator._tasks = [TaskModel(id="1", description="t1", status="pending")]
        assert orchestrator._is_done() is False

    def test_not_done_when_tests_failed(self, orchestrator):
        orchestrator._tasks = [TaskModel(id="1", description="t1", status="completed")]
        orchestrator._last_test_passed = False
        assert orchestrator._is_done() is False

    def test_not_done_when_any_task_failed(self, orchestrator):
        orchestrator._tasks = [TaskModel(id="1", description="t1", status="completed")]
        orchestrator._tasks.append(TaskModel(id="2", description="t2", status="failed"))
        orchestrator._last_test_passed = True
        assert orchestrator._is_done() is False

    def test_done_when_all_completed_and_tests_passed(self, orchestrator):
        orchestrator._tasks = [TaskModel(id="1", description="t1", status="completed")]
        orchestrator._last_test_passed = True
        assert orchestrator._is_done() is True

    def test_not_done_when_not_all_completed(self, orchestrator):
        orchestrator._tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="pending"),
        ]
        orchestrator._last_test_passed = True
        assert orchestrator._is_done() is False


class TestOrchestratorStatePersistence:
    """Verify state file is written with expected structure."""

    def test_save_state_writes_file(self, orchestrator, tmp_path):
        state_file = tmp_path / ".furrow_state.json"
        orchestrator.client.settings.state_file = state_file
        orchestrator._tasks = [TaskModel(id="1", description="d", status="completed")]
        orchestrator._last_test_passed = True
        orchestrator.cycles = 3

        orchestrator._save_state(final=False)

        content = json.loads(state_file.read_text())
        assert content["cycles"] == 3
        assert content["last_test_passed"] is True
        assert content["complete"] is True
        assert len(content["tasks"]) == 1
        assert content["tasks"][0]["id"] == "1"

    def test_save_state_final_flag(self, orchestrator, tmp_path):
        state_file = tmp_path / ".furrow_state.json"
        orchestrator.client.settings.state_file = state_file
        orchestrator._tasks = []
        orchestrator.cycles = 1

        orchestrator._save_state(final=True)

        content = json.loads(state_file.read_text())
        assert content["complete"] is True

    def test_save_state_swallows_errors(self, orchestrator):
        """State persistence is best-effort; must not raise."""
        orchestrator.client.settings.state_file = Path("/nonexistent_dir/state.json")
        orchestrator._save_state()  # should not raise


class TestOrchestratorMessageCallback:
    """Verify the message_callback is stored and produces output."""

    def test_message_callback_stored(self, mock_client):
        received = []

        async def cb(msg: str):
            received.append(msg)

        orch = Orchestrator(goal="g", client=mock_client, message_callback=cb)
        assert orch.message_callback is cb

    def test_emit_calls_callback(self, orchestrator):
        received = []

        async def cb(msg: str):
            received.append(msg)

        orchestrator.message_callback = cb

        async def run_test():
            await orchestrator._emit("hello")

        asyncio.run(run_test())
        assert len(received) == 1
        assert "hello" in received[0]


class TestOrchestratorRunLoop:
    """Integration tests for the full run loop with mocked subagents."""

    @pytest.mark.asyncio
    async def test_loop_stops_when_goal_complete(self, mock_client):
        """When a plan has one task, worker completes it, tests pass → loop stops."""
        from furrow.agents.planner import PlannerAgent
        from furrow.agents.tester import TesterAgent
        from furrow.agents.worker import WorkerAgent

        plan = Plan(
            tasks=[TaskModel(id="1", description="Add feature X", status="pending")],
            rationale="Add feature X",
        )
        test_result = TestResult(passed=True, summary="All tests passed")

        with patch.object(PlannerAgent, "plan", new_callable=AsyncMock, return_value=plan), \
             patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="Implemented feature X"), \
             patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=test_result):
            orch = Orchestrator(goal="Add feature X", client=mock_client)
            await orch.run()

        # Should have run exactly 1 cycle
        assert orch.cycles == 1
        assert orch._tasks[0].status == "completed"
        assert orch._last_test_passed is True

    @pytest.mark.asyncio
    async def test_loop_continues_on_test_failure(self, mock_client):
        """When tests fail, the loop should continue to at least a fix cycle."""
        from furrow.agents.planner import PlannerAgent
        from furrow.agents.tester import TesterAgent
        from furrow.agents.worker import WorkerAgent

        plan = Plan(
            tasks=[TaskModel(id="1", description="Add feature X", status="pending")],
            rationale="Add feature X",
        )
        test_result = TestResult(passed=False, summary="1 test failed", failures=["Test failed on line 42"])

        with patch.object(PlannerAgent, "plan", new_callable=AsyncMock, return_value=plan), \
             patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="Fixed"), \
             patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=test_result):
            orch = Orchestrator(goal="Add feature X", client=mock_client)
            # Set max_cycles to 3 to prevent infinite loop
            orch.client.settings.max_cycles = 3
            await orch.run()

        # Loop should have continued past the first cycle due to test failure
        assert orch.cycles >= 2

    @pytest.mark.asyncio
    async def test_max_cycles_limits_runs(self, mock_client):
        """When max_cycles is set, the loop should stop after that many cycles."""
        from furrow.agents.planner import PlannerAgent
        from furrow.agents.tester import TesterAgent
        from furrow.agents.worker import WorkerAgent

        plan = Plan(
            tasks=[TaskModel(id="1", description="Task", status="pending")],
            rationale="A task",
        )
        # Always fail tests so the loop would continue otherwise
        test_result = TestResult(passed=False, summary="fail", failures=["err"])

        orch = Orchestrator(goal="Test", client=mock_client)
        orch.client.settings.max_cycles = 2

        with patch.object(PlannerAgent, "plan", new_callable=AsyncMock, return_value=plan), \
             patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="done"), \
             patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=test_result):
            await orch.run()

        assert orch.cycles == 2


class TestOrchestratorSemaphore:
    """Verify max_parallel_tasks limits concurrency."""

    def test_semaphore_initialized(self, orchestrator):
        # The semaphore should be created with the configured limit
        assert orchestrator._semaphore is not None
        # With max_parallel_tasks=5 (default), the semaphore value should be 5
        assert orchestrator._semaphore._value == 5

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, mock_client):
        """Verify that only max_parallel_tasks workers run concurrently."""
        from furrow.agents.worker import WorkerAgent
        from furrow.config import TaskModel

        # Set max_parallel_tasks to 2
        mock_client.settings.max_parallel_tasks = 2

        plan = Plan(
            tasks=[
                TaskModel(id="1", description="t1"),
                TaskModel(id="2", description="t2"),
                TaskModel(id="3", description="t3"),
                TaskModel(id="4", description="t4"),
            ],
            rationale="test",
        )

        # Track concurrent workers
        current = 0
        peak = 0

        async def slow_run(self):
            nonlocal current, peak
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0.05)
            current -= 1
            return "done"

        orch = Orchestrator(goal="test", client=mock_client)
        with patch.object(WorkerAgent, "run", new=slow_run):
            # Replicate the _cycle worker execution logic
            worker_runs = [orch._run_worker(task) for task in plan.tasks]
            await asyncio.gather(*worker_runs)

        assert peak == 2  # Should never exceed max_parallel_tasks


# ---------------------------------------------------------------------------
# LLM Client — Ollama support
# ---------------------------------------------------------------------------

class TestOllamaSupport:
    """Verify Ollama provider is wired into the LLM client."""

    def test_ollama_config(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.OLLAMA

    @pytest.mark.asyncio
    async def test_complete_ollama_calls_api(self):
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434", ollama_model="test")
        client = LLMClient(settings=settings)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "generated text"}
        mock_resp.raise_for_status.return_value = None

        mock_httpx = MagicMock()
        mock_httpx.post = AsyncMock(return_value=mock_resp)
        with patch.object(type(client), "httpx_client", new_callable=PropertyMock, return_value=mock_httpx):
            result = await client._complete_ollama("hello", "system", "test")

        assert result == "generated text"
        mock_httpx.post.assert_awaited_once()
        call_args = mock_httpx.post.await_args
        assert "/api/generate" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_complete_routes_to_ollama(self):
        """Complete() should route to _complete_ollama when provider is OLLAMA."""
        settings = Settings(provider=Provider.OLLAMA, ollama_model="test")
        client = LLMClient(settings=settings)

        with patch.object(LLMClient, "_complete_ollama", new_callable=AsyncMock, return_value="ollama result") as mock_ollama:
            result = await client.complete("prompt", "system", "test")

        assert result == "ollama result"
        mock_ollama.assert_awaited_once_with("prompt", "system", "test")


# ---------------------------------------------------------------------------
# Tester improvements
# ---------------------------------------------------------------------------

class TestTesterImprovements:
    """Verify tester fixes: return code capture, exception handling."""

    def test_runner_output_includes_exit_code(self):
        """The test output should include the exit code from the process."""
        import inspect

        from furrow.agents.tester import TesterAgent
        source = inspect.getsource(TesterAgent._run_tests)
        assert "returncode" in source or "exit code" in source

    def test_no_redundant_exception_catch(self):
        """The tester should not have the redundant (FileNotFoundError, Exception) pattern."""
        import inspect

        from furrow.agents.tester import TesterAgent
        source = inspect.getsource(TesterAgent._run_tests)
        assert "FileNotFoundError" not in source

    def test_tester_has_retry_decorator(self):
        """The _complete_and_parse helper should have a tenacity retry decorator."""
        import inspect

        from furrow.agents.tester import TesterAgent
        source = inspect.getsource(TesterAgent._complete_and_parse)
        assert "retry" in source.lower()
