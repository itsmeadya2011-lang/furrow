"""Tests for the Orchestrator — loop logic, done-checking, dependencies, goal preservation."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator

from tests.conftest import FakeLLMClient, fake_settings


# --- _get_tasks and _is_done ---


class TestTaskTracking:
    def test_get_tasks_returns_empty_initially(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        assert orch._get_tasks() == []

    async def test_tasks_stored_after_cycle(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        tasks = [TaskModel(id="1", description="t1")]
        with (
            patch.object(orch.planner, "plan", new_callable=AsyncMock) as mock_plan,
            patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
            patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
        ):
            mock_plan.return_value = Plan(tasks=tasks, rationale="r")
            mock_worker_cls.return_value.run = AsyncMock(return_value="done")
            mock_tester_cls.return_value.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            await orch._cycle()
        assert orch._get_tasks() is not None
        assert len(orch._get_tasks()) == 1
        assert orch._get_tasks()[0].id == "1"


class TestIsDone:
    def test_is_done_no_test_result_all_tasks_completed(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        orch.tasks = [TaskModel(id="1", description="t1", status="completed")]
        # No test_result set yet
        assert orch._is_done() is True

    def test_is_done_no_test_result_with_pending(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        orch.tasks = [TaskModel(id="1", description="t1", status="pending")]
        assert orch._is_done() is False

    def test_is_done_no_test_result_with_failed(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        orch.tasks = [TaskModel(id="1", description="t1", status="failed")]
        assert orch._is_done() is False

    def test_is_done_tests_passed(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        orch.test_result = TestResult(passed=True, summary="all good", failures=[])
        assert orch._is_done() is True

    def test_is_done_tests_failed(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        orch.test_result = TestResult(passed=False, summary="broken", failures=["f1"])
        assert orch._is_done() is False

    def test_is_done_empty_tasks(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        orch.tasks = []
        assert orch._is_done() is True

    def test_is_done_mixed_status_completed(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        orch.tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        assert orch._is_done() is True


# --- _resolve_dependencies ---


class TestResolveDependencies:
    def test_no_dependencies_no_change(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        tasks = [
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b"),
        ]
        # Simulate all completed
        for t in tasks:
            t.status = "completed"
        orch._resolve_dependencies(tasks)
        assert all(t.status == "completed" for t in tasks)

    def test_dependency_on_failed_marks_dependent_failed(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        tasks = [
            TaskModel(id="1", description="a", dependencies=["2"]),
            TaskModel(id="2", description="b"),
        ]
        # Task 2 failed
        tasks[1].status = "failed"
        tasks[1].result = "error"
        # Task 1 completed
        tasks[0].status = "completed"
        tasks[0].result = "ok"

        orch._resolve_dependencies(tasks)

        assert tasks[0].status == "failed"
        assert "dependency" in (tasks[0].result or "")
        assert tasks[1].status == "failed"

    def test_dependency_on_completed_no_change(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        tasks = [
            TaskModel(id="1", description="a", dependencies=["2"]),
            TaskModel(id="2", description="b"),
        ]
        for t in tasks:
            t.status = "completed"
            t.result = "ok"

        orch._resolve_dependencies(tasks)

        assert all(t.status == "completed" for t in tasks)

    def test_nonexistent_dependency_ignored(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        tasks = [
            TaskModel(id="1", description="a", dependencies=["99"]),
        ]
        tasks[0].status = "completed"
        orch._resolve_dependencies(tasks)
        assert tasks[0].status == "completed"


# --- _planning_input ---


class TestPlanningInput:
    def test_returns_original_goal_when_no_fix_context(self, fake_client):
        orch = Orchestrator(goal="build a web server", client=fake_client)
        assert orch._planning_input() == "build a web server"

    def test_includes_fix_context_when_tests_failed(self, fake_client):
        orch = Orchestrator(goal="build a web server", client=fake_client)
        orch._fix_context = "test_a failed\nassertion error"
        result = orch._planning_input()
        assert "build a web server" in result
        assert "test_a failed" in result
        assert "Previous failures to fix" in result

    def test_does_not_include_fix_context_when_empty(self, fake_client):
        orch = Orchestrator(goal="goal", client=fake_client)
        # _fix_context defaults to ""
        assert orch._planning_input() == "goal"


# --- Original goal preservation ---


class TestGoalPreservation:
    def test_original_goal_not_mutated(self, fake_client):
        orch = Orchestrator(goal="original goal", client=fake_client)
        orch._fix_context = "fix this"
        assert orch.original_goal == "original goal"
        assert orch.goal == "original goal"

    async def test_goal_not_overwritten_on_test_failure(self, fake_client):
        orch = Orchestrator(goal="write a function", client=fake_client)
        # Simulate what happens in _cycle when tests fail
        with (
            patch.object(orch.planner, "plan", new_callable=AsyncMock) as mock_plan,
            patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
            patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
        ):
            mock_plan.return_value = Plan(
                tasks=[TaskModel(id="1", description="t1")], rationale="r"
            )
            mock_worker_cls.return_value.run = AsyncMock(return_value="done")
            mock_tester_cls.return_value.run = AsyncMock(
                return_value=TestResult(passed=False, summary="broken", failures=["f1"])
            )
            await orch._cycle()

        # Goal should still be the original
        assert orch.original_goal == "write a function"
        assert orch.goal == "write a function"
        assert orch._fix_context == "f1"

        # _planning_input should include both original goal and fix context
        input_text = orch._planning_input()
        assert "write a function" in input_text
        assert "f1" in input_text


# --- max_cycles ---


class TestMaxCycles:
    async def test_max_cycles_stops_loop(self, fake_client):
        """max_cycles limits the number of cycles the loop runs."""
        from furrow.config import Settings, Provider

        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t", max_cycles=2)
        client = FakeLLMClient(settings=settings)
        orch = Orchestrator(goal="test", client=client)

        with (
            patch.object(orch.planner, "plan", new_callable=AsyncMock) as mock_plan,
            patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
            patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
        ):
            mock_plan.return_value = Plan(
                tasks=[TaskModel(id="1", description="t")], rationale="r"
            )
            mock_worker_cls.return_value.run = AsyncMock(return_value="done")
            mock_tester_cls.return_value.run = AsyncMock(
                return_value=TestResult(passed=False, summary="fail", failures=["e"])
            )
            await orch.run()

        assert orch.cycles == 2

    async def test_max_cycles_zero_runs_until_done(self, fake_client):
        """max_cycles=0 means no limit, loop until test_result passes."""
        from furrow.config import Settings, Provider

        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t", max_cycles=0)
        client = FakeLLMClient(settings=settings)
        orch = Orchestrator(goal="test", client=client)

        call_count = 0

        async def mock_plan(goal):
            nonlocal call_count
            call_count += 1
            return Plan(tasks=[TaskModel(id="1", description="t")], rationale="r")

        with (
            patch.object(orch.planner, "plan", side_effect=mock_plan),
            patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
            patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
        ):
            mock_worker_cls.return_value.run = AsyncMock(return_value="done")
            mock_tester_cls.return_value.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            await orch.run()

        assert call_count == 1
        assert orch.cycles == 1


# --- max_parallel_tasks semaphore ---


class TestParallelLimit:
    async def test_semaphore_limits_concurrency(self, tmp_path):
        """Verify that max_parallel_tasks is respected by the semaphore."""
        from furrow.config import Settings, Provider

        settings = Settings(
            provider=Provider.ANTHROPIC, anthropic_api_key="t", max_parallel_tasks=2, max_cycles=1
        )
        client = FakeLLMClient(settings=settings)
        orch = Orchestrator(goal="test", client=client)

        tasks = [TaskModel(id=str(i), description=f"task {i}") for i in range(5)]

        max_concurrent = 0
        current_concurrent = 0

        with (
            patch.object(orch.planner, "plan", new_callable=AsyncMock) as mock_plan,
            patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
            patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
        ):
            mock_plan.return_value = Plan(tasks=tasks, rationale="r")

            async def tracking_run(self_task):
                nonlocal max_concurrent, current_concurrent
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.01)
                current_concurrent -= 1
                return "done"

            mock_worker_cls.return_value.run = tracking_run
            mock_tester_cls.return_value.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            await orch.run()

        assert max_concurrent <= 2


# --- message_queue / _log ---


class TestMessageQueue:
    async def test_log_sends_to_queue(self, fake_client):
        queue: asyncio.Queue[str] = asyncio.Queue()
        orch = Orchestrator(goal="test", client=fake_client, message_queue=queue)
        orch._log("hello world")
        msg = queue.get_nowait()
        assert "hello world" in msg

    async def test_log_strips_rich_markup(self, fake_client):
        queue: asyncio.Queue[str] = asyncio.Queue()
        orch = Orchestrator(goal="test", client=fake_client, message_queue=queue)
        orch._log("[yellow]Stripping tags[/yellow]")
        msg = queue.get_nowait()
        assert "[yellow]" not in msg
        assert "Stripping tags" in msg

    async def test_log_without_queue_does_not_error(self, fake_client):
        orch = Orchestrator(goal="test", client=fake_client)
        orch._log("just console")  # should not raise

    async def test_log_with_style_sends_plain_to_queue(self, fake_client):
        queue: asyncio.Queue[str] = asyncio.Queue()
        orch = Orchestrator(goal="test", client=fake_client, message_queue=queue)
        orch._log("styled message", style="bold green")
        msg = queue.get_nowait()
        assert "styled message" in msg


# --- Full run loop ---


class TestRunLoop:
    async def test_run_stops_when_tests_pass(self, fake_client):
        orch = Orchestrator(goal="test goal", client=fake_client)
        with (
            patch.object(orch.planner, "plan", new_callable=AsyncMock) as mock_plan,
            patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
            patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
        ):
            mock_plan.return_value = Plan(
                tasks=[TaskModel(id="1", description="t1")], rationale="r"
            )
            mock_worker_cls.return_value.run = AsyncMock(return_value="done")
            mock_tester_cls.return_value.run = AsyncMock(
                return_value=TestResult(passed=True, summary="all pass", failures=[])
            )
            await orch.run()

        assert orch.cycles == 1
        assert orch.original_goal == "test goal"

    async def test_run_continues_when_tests_fail(self, fake_client):
        from furrow.config import Settings, Provider

        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t", max_cycles=2)
        client = FakeLLMClient(settings=settings)
        orch = Orchestrator(goal="test", client=client)

        plan_calls = 0

        async def mock_plan_side(goal):
            nonlocal plan_calls
            plan_calls += 1
            return Plan(tasks=[TaskModel(id="1", description="t")], rationale="r")

        with (
            patch.object(orch.planner, "plan", side_effect=mock_plan_side),
            patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
            patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
        ):
            mock_worker_cls.return_value.run = AsyncMock(return_value="done")
            # First test fails, second passes
            test_results = [
                TestResult(passed=False, summary="fail", failures=["err1"]),
                TestResult(passed=True, summary="pass", failures=[]),
            ]
            mock_tester_cls.return_value.run = AsyncMock(side_effect=test_results)
            await orch.run()

        assert orch.cycles == 2
        assert plan_calls == 2  # Planner called once per cycle

    async def test_run_no_tasks_stops(self, fake_client):
        orch = Orchestrator(goal="done", client=fake_client)
        with (
            patch.object(orch.planner, "plan", new_callable=AsyncMock) as mock_plan,
        ):
            mock_plan.return_value = Plan(tasks=[], rationale="nothing to do")
            await orch.run()

        assert orch.cycles == 1

    async def test_tester_runs_against_original_goal(self, fake_client):
        from furrow.config import Settings, Provider
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t", max_cycles=1)
        client = FakeLLMClient(settings=settings)
        orch = Orchestrator(goal="original", client=client)
        with (
            patch.object(orch.planner, "plan", new_callable=AsyncMock) as mock_plan,
            patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
            patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
        ):
            mock_plan.return_value = Plan(
                tasks=[TaskModel(id="1", description="t")], rationale="r"
            )
            mock_worker_cls.return_value.run = AsyncMock(return_value="done")
            mock_tester_cls.return_value = MagicMock()
            mock_tester_cls.return_value.run = AsyncMock(
                return_value=TestResult(passed=False, summary="x", failures=["y"])
            )
            await orch.run()

        # Even after cycle 1 (tests fail), _fix_context is set
        # The tester.run should have been called with "original"
        mock_tester_cls.return_value.run.assert_awaited()
        call_args = mock_tester_cls.return_value.run.call_args
        assert call_args[0][0] == "original"  # goal argument
