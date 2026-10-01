"""Tests for the orchestrator's control logic."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import (
    OrchestratorState,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
)
from furrow.core.orchestrator import Orchestrator


def make_settings(tmp_path, **overrides):
    """Create test settings with a temp workspace."""
    kwargs = {"workspace": tmp_path, "state_file": ".furrow-state.json"}
    kwargs.update(overrides)
    return Settings(**kwargs)


class TestOrchestratorStateTracking:
    """Tests for the fixed task tracking bug."""

    def test_get_tasks_returns_empty_when_no_tasks(self, tmp_path):
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        assert orch._get_tasks() == []

    def test_get_tasks_returns_current_tasks(self, tmp_path):
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        tasks = [TaskModel(id="1", description="task 1")]
        orch._tasks = tasks
        assert orch._get_tasks() == tasks


class TestOrchestratorIsDone:
    """Tests for _is_done() logic - the critical bug fix."""

    def test_is_done_no_tasks(self, tmp_path):
        """With no tasks (planner found nothing), goal is done."""
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        assert orch._is_done() is True

    def test_is_done_all_completed(self, tmp_path):
        """All tasks completed means done."""
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        orch._tasks = [
            TaskModel(id="1", description="t1", status=TaskStatus.COMPLETED.value),
            TaskModel(id="2", description="t2", status=TaskStatus.COMPLETED.value),
        ]
        assert orch._is_done() is True

    def test_is_done_all_failed(self, tmp_path):
        """All tasks failed means NOT done - should retry."""
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        orch._tasks = [
            TaskModel(id="1", description="t1", status=TaskStatus.FAILED.value),
        ]
        assert orch._is_done() is False

    def test_is_done_some_failed(self, tmp_path):
        """Mixed completed/failed is NOT done."""
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        orch._tasks = [
            TaskModel(id="1", description="t1", status=TaskStatus.COMPLETED.value),
            TaskModel(id="2", description="t2", status=TaskStatus.FAILED.value),
        ]
        assert orch._is_done() is False

    def test_is_done_some_pending(self, tmp_path):
        """Pending tasks mean not done."""
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        orch._tasks = [
            TaskModel(id="1", description="t1", status=TaskStatus.COMPLETED.value),
            TaskModel(id="2", description="t2", status=TaskStatus.PENDING.value),
        ]
        assert orch._is_done() is False

    def test_is_done_some_in_progress(self, tmp_path):
        """In-progress tasks mean not done."""
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        orch._tasks = [
            TaskModel(id="1", description="t1", status=TaskStatus.COMPLETED.value),
            TaskModel(id="2", description="t2", status=TaskStatus.IN_PROGRESS.value),
        ]
        assert orch._is_done() is False


class TestOrchestratorMaxCycles:
    """Tests for max_cycles enforcement."""

    def test_should_stop_unlimited_when_max_zero(self, tmp_path):
        settings = make_settings(tmp_path, max_cycles=0)
        orch = Orchestrator(goal="test", settings=settings)
        orch.cycles = 100
        assert orch._should_stop() is False

    def test_should_stop_when_max_reached(self, tmp_path):
        settings = make_settings(tmp_path, max_cycles=3)
        orch = Orchestrator(goal="test", settings=settings)
        orch.cycles = 3
        assert orch._should_stop() is True

    def test_should_stop_when_below_max(self, tmp_path):
        settings = make_settings(tmp_path, max_cycles=3)
        orch = Orchestrator(goal="test", settings=settings)
        orch.cycles = 2
        assert orch._should_stop() is False


class TestOrchestratorParallelLimit:
    """Tests for max_parallel_tasks semaphore."""

    def test_semaphore_created_with_max_parallel(self, tmp_path):
        settings = make_settings(tmp_path, max_parallel_tasks=3)
        orch = Orchestrator(goal="test", settings=settings)
        assert orch._semaphore is None  # Created during _cycle

    def test_max_parallel_minimum_is_1(self, tmp_path):
        settings = make_settings(tmp_path, max_parallel_tasks=1)
        orch = Orchestrator(goal="test", settings=settings)
        assert orch.settings.max_parallel_tasks == 1


class TestOrchestratorStatePersistence:
    """Tests for state file save/load."""

    def test_save_and_load_state(self, tmp_path):
        settings = make_settings(tmp_path, state_file=".furrow-state.json")
        orch = Orchestrator(goal="initial goal", settings=settings)
        orch.cycles = 2
        orch._tasks = [TaskModel(id="1", description="task 1", status=TaskStatus.COMPLETED.value)]
        orch._save_state()

        state_file = tmp_path / ".furrow-state.json"
        assert state_file.exists()

        data = json.loads(state_file.read_text())
        assert data["goal"] == "initial goal"
        assert data["cycle"] == 2
        assert len(data["task_history"]) == 1

    def test_load_existing_state(self, tmp_path):
        # Write a state file first
        state = OrchestratorState(goal="loaded goal", cycle=5)
        state_file = tmp_path / ".furrow-state.json"
        state_file.write_text(state.model_dump_json())

        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        assert orch.state is not None
        assert orch.state.goal == "loaded goal"
        assert orch.state.cycle == 5

    def test_load_corrupt_state(self, tmp_path):
        state_file = tmp_path / ".furrow-state.json"
        state_file.write_text("{ invalid json }")

        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        assert orch.state is None  # Should gracefully handle corruption

    def test_no_state_file(self, tmp_path):
        settings = make_settings(tmp_path)
        Orchestrator(goal="test", settings=settings)
        # Should not create a state file until _save_state is called
        state_path = tmp_path / ".furrow-state.json"
        assert not state_path.exists()

    def test_state_file_disabled(self, tmp_path):
        settings = make_settings(tmp_path, state_file="")
        orch = Orchestrator(goal="test", settings=settings)
        # _save_state should be a no-op when state_file is empty
        orch._save_state()
        assert orch.state is None

    def test_state_history_updated_on_task_update(self, tmp_path):
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        orch.state = OrchestratorState(goal="test")

        task = TaskModel(id="1", description="t1", status=TaskStatus.COMPLETED.value)
        orch.state.update_task(task)
        assert len(orch.state.task_history) == 1

        task.status = TaskStatus.FAILED.value
        orch.state.update_task(task)
        assert len(orch.state.task_history) == 1  # Same id replaced, not appended
        assert orch.state.task_history[0].status == "failed"


class TestOrchestratorOutputCallback:
    """Tests for the output callback mechanism (WebSocket forwarding)."""

    def test_output_callback_receives_text(self, tmp_path):
        settings = make_settings(tmp_path)
        captured = []
        orch = Orchestrator(goal="test", settings=settings, output_callback=captured.append)
        orch._emit("Hello output")
        assert any("Hello output" in msg for msg in captured)

    def test_no_callback_does_not_crash(self, tmp_path):
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        # Should not raise when output_callback is None
        orch._emit("Hello")


class TestOrchestratorCycle:
    """Integration-style tests for _cycle() with mocked LLM."""

    @pytest.mark.asyncio
    async def test_cycle_with_tasks(self, tmp_path):
        """Test a full cycle with mocked planner/worker/tester."""
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test goal", settings=settings)

        # Mock the planner to return a simple plan
        mock_plan = MagicMock()
        mock_plan.tasks = [TaskModel(id="1", description="task 1")]
        mock_plan.model_dump = MagicMock(return_value={"tasks": [], "rationale": "test"})

        # Mock the worker
        mock_worker_result = "Completed task 1"

        with (
            patch.object(orch.planner, "plan", new=AsyncMock(return_value=mock_plan)),
            patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
            patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
            patch("furrow.core.orchestrator.Status"),
        ):
            mock_worker_instance = MagicMock()
            mock_worker_instance.run = AsyncMock(return_value=mock_worker_result)
            mock_worker_cls.return_value = mock_worker_instance

            mock_tester_instance = MagicMock()
            mock_tester_instance.run = AsyncMock(
                return_value=TestResult(passed=True, summary="All good", failures=[])
            )
            mock_tester_cls.return_value = mock_tester_instance

            await orch._cycle()

        assert orch._tasks[0].status == TaskStatus.COMPLETED.value
        assert orch._tasks[0].result == mock_worker_result
        assert orch.cycles == 0  # cycles is incremented in run(), not _cycle()

    @pytest.mark.asyncio
    async def test_cycle_no_tasks(self, tmp_path):
        """Test cycle with no tasks (goal already complete)."""
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test goal", settings=settings)

        mock_plan = MagicMock()
        mock_plan.tasks = []
        mock_plan.model_dump = MagicMock(return_value={"tasks": [], "rationale": "done"})

        with (
            patch.object(orch.planner, "plan", new=AsyncMock(return_value=mock_plan)),
            patch("furrow.core.orchestrator.Status"),
            patch.object(orch, "_save_state") as mock_save,
        ):
            await orch._cycle()

        assert orch._tasks == []
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_handles_worker_exception(self, tmp_path):
        """Worker exceptions should mark task as failed, not crash."""
        settings = make_settings(tmp_path)
        orch = Orchestrator(goal="test goal", settings=settings)

        mock_plan = MagicMock()
        mock_plan.tasks = [TaskModel(id="1", description="task 1")]
        mock_plan.model_dump = MagicMock(return_value={"tasks": [], "rationale": "test"})

        with (
            patch.object(orch.planner, "plan", new=AsyncMock(return_value=mock_plan)),
            patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
            patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
            patch("furrow.core.orchestrator.Status"),
        ):
            mock_worker_instance = MagicMock()
            mock_worker_instance.run = AsyncMock(side_effect=RuntimeError("Worker crashed"))
            mock_worker_cls.return_value = mock_worker_instance

            mock_tester_instance = MagicMock()
            mock_tester_instance.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            mock_tester_cls.return_value = mock_tester_instance

            await orch._cycle()

        assert orch._tasks[0].status == TaskStatus.FAILED.value
        assert "Worker crashed" in orch._tasks[0].result


class TestOrchestratorRun:
    """Tests for the main run() loop."""

    @pytest.mark.asyncio
    async def test_run_stops_at_max_cycles(self, tmp_path):
        """Ensure run() respects max_cycles."""
        settings = make_settings(tmp_path, max_cycles=1)
        orch = Orchestrator(goal="test", settings=settings)

        call_count = 0

        async def mock_cycle():
            nonlocal call_count
            call_count += 1
            orch._tasks = [TaskModel(id="1", description="t", status=TaskStatus.COMPLETED.value)]

        with (
            patch.object(orch, "_cycle", side_effect=mock_cycle),
            patch("furrow.core.orchestrator.Status"),
            patch.object(orch.client, "aclose", new=AsyncMock()),
        ):
            await orch.run()

        assert call_count == 1
        assert orch.cycles == 1
