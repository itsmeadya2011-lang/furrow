from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import TaskModel
from furrow.core.orchestrator import Orchestrator


class TestIsDone:
    """Tests for Orchestrator._is_done() logic."""

    def test_empty_tasks_returns_true(self):
        orch = Orchestrator(goal="test", on_event=None)
        orch.tasks = []
        assert orch._is_done() is True

    def test_all_completed_returns_true(self):
        orch = Orchestrator(goal="test", on_event=None)
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        assert orch._is_done() is True

    def test_any_failed_returns_false(self):
        orch = Orchestrator(goal="test", on_event=None)
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ]
        assert orch._is_done() is False

    def test_mix_completed_and_pending_returns_false(self):
        orch = Orchestrator(goal="test", on_event=None)
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="pending"),
        ]
        assert orch._is_done() is False

    def test_all_pending_returns_false(self):
        orch = Orchestrator(goal="test", on_event=None)
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="pending"),
            TaskModel(id="2", description="task 2", status="pending"),
        ]
        assert orch._is_done() is False


class TestEmit:
    """Tests for Orchestrator._emit() callback."""

    @pytest.mark.asyncio
    async def test_emit_calls_callback(self):
        events = []
        async def on_event(msg):
            events.append(msg)

        orch = Orchestrator(goal="test", on_event=on_event)
        await orch._emit("test message")

        assert len(events) == 1
        assert events[0] == "test message"

    @pytest.mark.asyncio
    async def test_emit_works_without_callback(self):
        orch = Orchestrator(goal="test", on_event=None)
        # Should not raise
        await orch._emit("test message")


class TestExecuteTasks:
    """Tests for Orchestrator._execute_tasks() topological execution."""

    @pytest.mark.asyncio
    async def test_no_dependencies_all_run_in_one_batch(self):
        orch = Orchestrator(goal="test", on_event=None)
        tasks = [
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
            TaskModel(id="3", description="task 3"),
        ]

        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = "done"
            mock_worker.return_value = mock_instance

            result = await orch._execute_tasks(tasks)

        assert mock_worker.call_count == 3
        for task in result:
            assert task.status == "completed"
            assert task.result == "done"

    @pytest.mark.asyncio
    async def test_chained_dependencies_1_to_2_to_3(self):
        orch = Orchestrator(goal="test", on_event=None)
        tasks = [
            TaskModel(id="1", description="task 1", dependencies=[]),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
            TaskModel(id="3", description="task 3", dependencies=["2"]),
        ]

        run_order = []

        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker:
            def create_mock(task=None, client=None):
                mock = AsyncMock()
                async def run_wrapper():
                    run_order.append(task.id)
                    return "done"
                mock.run = run_wrapper
                return mock
            mock_worker.side_effect = create_mock

            result = await orch._execute_tasks(tasks)

        assert run_order == ["1", "2", "3"]
        for task in result:
            assert task.status == "completed"

    @pytest.mark.asyncio
    async def test_diamond_dependencies(self):
        orch = Orchestrator(goal="test", on_event=None)
        tasks = [
            TaskModel(id="1", description="task 1", dependencies=[]),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
            TaskModel(id="3", description="task 3", dependencies=["1"]),
            TaskModel(id="4", description="task 4", dependencies=["2", "3"]),
        ]

        run_order = []

        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker:
            def create_mock(task=None, client=None):
                mock = AsyncMock()
                async def run_wrapper():
                    run_order.append(task.id)
                    return "done"
                mock.run = run_wrapper
                return mock
            mock_worker.side_effect = create_mock

            result = await orch._execute_tasks(tasks)

        assert run_order[0] == "1"
        assert set(run_order[1:3]) == {"2", "3"}
        assert run_order[3] == "4"
        for task in result:
            assert task.status == "completed"

    @pytest.mark.asyncio
    async def test_circular_dependencies_does_not_hang(self):
        orch = Orchestrator(goal="test", on_event=None)
        tasks = [
            TaskModel(id="1", description="task 1", dependencies=["2"]),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
        ]

        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = "done"
            mock_worker.return_value = mock_instance

            result = await orch._execute_tasks(tasks)

        # Neither task should run because of circular dependency
        assert mock_worker.call_count == 0
        for task in result:
            assert task.status == "pending"

    @pytest.mark.asyncio
    async def test_non_existent_dependency_handled_gracefully(self):
        orch = Orchestrator(goal="test", on_event=None)
        tasks = [
            TaskModel(id="1", description="task 1", dependencies=["nonexistent"]),
            TaskModel(id="2", description="task 2", dependencies=[]),
        ]

        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = "done"
            mock_worker.return_value = mock_instance

            result = await orch._execute_tasks(tasks)

        # Task 2 should run (no deps), task 1 should not (depends on nonexistent)
        assert mock_worker.call_count == 1
        task1 = next(t for t in result if t.id == "1")
        task2 = next(t for t in result if t.id == "2")
        assert task1.status == "pending"
        assert task2.status == "completed"


class TestMaxCycles:
    """Tests for max_cycles limit in Orchestrator.run()."""

    @pytest.mark.asyncio
    async def test_max_cycles_respected(self):
        events = []
        async def on_event(msg):
            events.append(msg)

        # Create orchestrator with max_cycles=1 via settings
        from furrow.config import settings
        original_max = settings.max_cycles
        settings.max_cycles = 1

        try:
            with patch("furrow.core.orchestrator.PlannerAgent") as mock_planner, \
                 patch("furrow.core.orchestrator.WorkerAgent") as mock_worker, \
                 patch("furrow.core.orchestrator.TesterAgent") as mock_tester:
                # Return a task
                mock_plan = MagicMock()
                mock_plan.tasks = [TaskModel(id="1", description="task 1", status="pending")]
                mock_instance = AsyncMock()
                mock_instance.plan.return_value = mock_plan
                mock_planner.return_value = mock_instance

                # Mock WorkerAgent to raise OSError so task fails (caught by orchestrator)
                mock_worker_instance = AsyncMock()
                mock_worker_instance.run.side_effect = OSError("Worker failed")
                mock_worker.return_value = mock_worker_instance

                # Mock TesterAgent to return failed test result
                mock_tester_instance = AsyncMock()
                from furrow.config import TestResult
                mock_tester_instance.run.return_value = TestResult(passed=False, summary="failed", failures=["test failed"])
                mock_tester.return_value = mock_tester_instance

                orch = Orchestrator(goal="test", on_event=on_event)
                await orch.run()

            # Should have run exactly 1 cycle
            assert orch.cycles == 1
            # Task failed, so _is_done() is False, but max_cycles=1 so it should stop with max cycles message
            assert any("Max cycles (1) reached" in str(e) for e in events)
        finally:
            settings.max_cycles = original_max

    @pytest.mark.asyncio
    async def test_loop_terminates_when_planner_returns_no_tasks(self):
        events = []
        async def on_event(msg):
            events.append(msg)

        with patch("furrow.core.orchestrator.PlannerAgent") as mock_planner:
            mock_instance = AsyncMock()
            mock_instance.plan.return_value = MagicMock(tasks=[])
            mock_planner.return_value = mock_instance

            orch = Orchestrator(goal="test", on_event=on_event)
            await orch.run()

        # Should have run 1 cycle and stopped
        assert orch.cycles == 1
        assert any("No tasks planned" in str(e) for e in events)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])