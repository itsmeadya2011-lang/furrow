from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_task(task_id: str, deps: list[str] | None = None) -> TaskModel:
    return TaskModel(id=task_id, description=f"task {task_id}", dependencies=deps or [])


@pytest.fixture
def orch() -> Orchestrator:
    client = MagicMock()
    settings = MagicMock()
    settings.max_cycles = 0
    settings.max_parallel_tasks = 5
    settings.max_tokens = 4096
    settings.provider = "anthropic"
    client.settings = settings
    orch = Orchestrator(goal="test goal", client=client)
    return orch


class TestIsDone:
    def test_empty_returns_true(self, orch: Orchestrator):
        orch._tasks = []
        assert orch._is_done() is True

    def test_all_completed_returns_true(self, orch: Orchestrator):
        t1 = _make_task("1")
        t1.status = "completed"
        t2 = _make_task("2")
        t2.status = "completed"
        orch._tasks = [t1, t2]
        assert orch._is_done() is True

    def test_any_failed_returns_false(self, orch: Orchestrator):
        t1 = _make_task("1")
        t1.status = "completed"
        t2 = _make_task("2")
        t2.status = "failed"
        orch._tasks = [t1, t2]
        assert orch._is_done() is False

    def test_partial_pending_returns_false(self, orch: Orchestrator):
        t1 = _make_task("1")
        t1.status = "completed"
        t2 = _make_task("2")
        orch._tasks = [t1, t2]
        assert orch._is_done() is False


class TestTaskRetrieval:
    def test_get_tasks_returns_instance_state(self, orch: Orchestrator):
        tasks = [_make_task("1"), _make_task("2")]
        orch._tasks = tasks
        assert orch._get_tasks() is tasks


class TestExecuteTasks:
    def test_independent_tasks_run_in_parallel(self, orch: Orchestrator):
        t1 = _make_task("1")
        t2 = _make_task("2")

        start_times: dict[str, float] = {}

        class MockWorker:
            def __init__(self, task, client=None):
                self.task = task

            async def run(self):
                start_times[self.task.id] = asyncio.get_event_loop().time()
                await asyncio.sleep(0.02)
                return f"result {self.task.id}"

        with patch("furrow.core.orchestrator.WorkerAgent", MockWorker):
            asyncio.run(orch._execute_tasks([t1, t2]))

        # Both should have started within a tiny window (parallel, not sequential)
        assert t1.status == "completed"
        assert t2.status == "completed"
        assert abs(start_times["1"] - start_times["2"]) < 0.015

    def test_dependency_ordering(self, orch: Orchestrator):
        t1 = _make_task("1", deps=[])
        t2 = _make_task("2", deps=["1"])

        start_order: list[str] = []

        class MockWorker:
            def __init__(self, task, client=None):
                self.task = task

            async def run(self):
                start_order.append(self.task.id)
                await asyncio.sleep(0.02)
                return f"result {self.task.id}"

        with patch("furrow.core.orchestrator.WorkerAgent", MockWorker):
            asyncio.run(orch._execute_tasks([t1, t2]))

        # t1 must start before t2
        assert start_order.index("1") < start_order.index("2")

    def test_diamond_dependency(self, orch: Orchestrator):
        #   t1
        #  /  \
        # t2   t3
        #  \  /
        #   t4
        t1 = _make_task("1", deps=[])
        t2 = _make_task("2", deps=["1"])
        t3 = _make_task("3", deps=["1"])
        t4 = _make_task("4", deps=["2", "3"])

        class MockWorker:
            def __init__(self, task, client=None):
                self.task = task

            async def run(self):
                return f"result {self.task.id}"

        with patch("furrow.core.orchestrator.WorkerAgent", MockWorker):
            asyncio.run(orch._execute_tasks([t1, t2, t3, t4]))

        assert all(t.status == "completed" for t in [t1, t2, t3, t4])

    def test_circular_dependency_does_not_hang(self, orch: Orchestrator):
        # t1 depends on t2, t2 depends on t1
        t1 = _make_task("1", deps=["2"])
        t2 = _make_task("2", deps=["1"])

        class MockWorker:
            def __init__(self, task, client=None):
                self.task = task

            async def run(self):
                return "result"

        with patch("furrow.core.orchestrator.WorkerAgent", MockWorker):
            asyncio.run(orch._execute_tasks([t1, t2]))

        # Both should be marked failed, not hang
        assert t1.status == "failed"
        assert t2.status == "failed"

    def test_worker_exception_marks_task_failed(self, orch: Orchestrator):
        t1 = _make_task("1")

        class MockWorker:
            def __init__(self, task, client=None):
                self.task = task

            async def run(self):
                raise RuntimeError("boom")

        with patch("furrow.core.orchestrator.WorkerAgent", MockWorker):
            asyncio.run(orch._execute_tasks([t1]))

        assert t1.status == "failed"
        assert "boom" in (t1.result or "")

    def test_max_parallel_tasks_limits_concurrency(self, orch: Orchestrator):
        orch.client.settings.max_parallel_tasks = 2
        tasks = [_make_task(str(i)) for i in range(5)]
        in_flight = 0
        max_concurrent = 0

        class MockWorker:
            def __init__(self, task, client=None):
                self.task = task

            async def run(self):
                nonlocal in_flight, max_concurrent
                in_flight += 1
                max_concurrent = max(max_concurrent, in_flight)
                await asyncio.sleep(0.02)
                in_flight -= 1
                return "result"

        with patch("furrow.core.orchestrator.WorkerAgent", MockWorker):
            asyncio.run(orch._execute_tasks(tasks))

        assert max_concurrent <= 2

    def test_task_exception_does_not_block_others(self, orch: Orchestrator):
        t1 = _make_task("1")
        t2 = _make_task("2")
        t3 = _make_task("3")

        class MockWorker:
            def __init__(self, task, client=None):
                self.task = task

            async def run(self):
                if self.task.id == "2":
                    raise ValueError("fail")
                return f"ok {self.task.id}"

        with patch("furrow.core.orchestrator.WorkerAgent", MockWorker):
            asyncio.run(orch._execute_tasks([t1, t2, t3]))

        assert t1.status == "completed"
        assert t2.status == "failed"
        assert t3.status == "completed"


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_completes_when_all_tasks_done(self, orch: Orchestrator):
        orch.client.settings.max_cycles = 0
        orch.client.settings.max_parallel_tasks = 5

        cycle = 0

        async def mock_plan(goal):
            nonlocal cycle
            cycle += 1
            if cycle == 1:
                return Plan(tasks=[_make_task("1")], rationale="first")
            return Plan(tasks=[], rationale="done")

        with patch.object(orch.planner, "plan", mock_plan):
            with patch("furrow.core.orchestrator.WorkerAgent") as mw:
                mw.return_value.run = AsyncMock(return_value="done")
                with patch("furrow.core.orchestrator.TesterAgent") as mt:
                    mt.return_value.run = AsyncMock(
                        return_value=TestResult(passed=True, summary="ok", failures=[])
                    )
                    await orch.run()

        assert orch.cycles == 2
        assert orch._tasks == []

    @pytest.mark.asyncio
    async def test_stops_at_max_cycles(self, orch: Orchestrator):
        orch.client.settings.max_cycles = 2

        async def mock_plan(goal):
            return Plan(tasks=[_make_task("1")], rationale="test")

        with patch.object(orch.planner, "plan", mock_plan):
            with patch("furrow.core.orchestrator.WorkerAgent") as mw:
                mw.return_value.run = AsyncMock(return_value="done")
                with patch("furrow.core.orchestrator.TesterAgent") as mt:
                    mt.return_value.run = AsyncMock(
                        return_value=TestResult(passed=True, summary="ok", failures=[])
                    )
                    await orch.run()

        assert orch.cycles == 2

    @pytest.mark.asyncio
    async def test_fixes_goal_on_test_failure(self, orch: Orchestrator):
        orch.client.settings.max_cycles = 0

        cycle = 0

        async def mock_plan(goal):
            nonlocal cycle
            cycle += 1
            if cycle == 1:
                return Plan(tasks=[_make_task("1")], rationale="test")
            return Plan(tasks=[], rationale="done")

        with patch.object(orch.planner, "plan", mock_plan):
            with patch("furrow.core.orchestrator.WorkerAgent") as mw:
                mw.return_value.run = AsyncMock(return_value="done")
                with patch("furrow.core.orchestrator.TesterAgent") as mt:
                    mt.return_value.run = AsyncMock(
                        side_effect=[
                            TestResult(passed=False, summary="fail", failures=["broken"]),
                            TestResult(passed=True, summary="ok", failures=[]),
                        ]
                    )
                    await orch.run()

        assert orch.cycles == 2

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_stops(self, orch: Orchestrator):
        async def mock_plan(goal):
            return Plan(tasks=[_make_task("1")], rationale="test")

        # Simulate infinite loop: plan always returns tasks, tests fail,
        # but we raise KeyboardInterrupt to simulate Ctrl+C
        mock_worker_run = AsyncMock(side_effect=KeyboardInterrupt())

        with patch.object(orch.planner, "plan", mock_plan):
            with patch("furrow.core.orchestrator.WorkerAgent") as mw:
                mw.return_value.run = mock_worker_run
                with pytest.raises(KeyboardInterrupt):
                    await orch._cycle()
