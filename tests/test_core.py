"""Tests for Furrow core components: orchestrator, config, and agents."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_plan() -> Plan:
    return Plan(
        tasks=[
            TaskModel(id="1", description="Implement auth module", files=["src/auth.py"]),
            TaskModel(id="2", description="Write auth tests", files=["tests/test_auth.py"]),
        ],
        rationale="Breaking auth into implementation and tests",
    )


@pytest.fixture
def sample_test_result_pass() -> TestResult:
    return TestResult(passed=True, summary="All 3 tests passed", failures=[])


@pytest.fixture
def sample_test_result_fail() -> TestResult:
    return TestResult(
        passed=False,
        summary="2 of 3 tests failed",
        failures=["test_auth.test_login failed: invalid token", "test_auth.test_logout failed: session not cleared"],
    )


@pytest.fixture
def mock_client() -> AsyncMock:
    """A mock LLMClient that returns valid JSON for planning."""
    client = AsyncMock()
    client.settings = Settings()

    # Default planner response: two tasks
    client.complete.return_value = json.dumps({
        "tasks": [
            {"id": "1", "description": "Task one", "files": [], "dependencies": []},
            {"id": "2", "description": "Task two", "files": [], "dependencies": []},
        ],
        "rationale": "Test plan",
    })
    return client


@pytest.fixture
def failing_test_client() -> AsyncMock:
    """A mock LLMClient that returns failing test output."""
    client = AsyncMock()
    client.settings = Settings()
    client.complete.side_effect = [
        # First call (planner) returns a plan
        json.dumps({
            "tasks": [{"id": "1", "description": "Fix something", "files": [], "dependencies": []}],
            "rationale": "Fix it",
        }),
    ]
    return client


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestProvider:
    def test_provider_values(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"


class TestTaskModel:
    def test_defaults(self):
        t = TaskModel(id="1", description="do thing")
        assert t.status == "pending"
        assert t.files == []
        assert t.dependencies == []
        assert t.result is None

    def test_custom_values(self):
        t = TaskModel(
            id="t1", description="test", files=["a.py"], dependencies=["t0"], status="completed", result="done"
        )
        assert t.id == "t1"
        assert t.status == "completed"
        assert t.result == "done"


class TestPlan:
    def test_plan_with_tasks(self, sample_plan):
        assert len(sample_plan.tasks) == 2
        assert sample_plan.tasks[0].description == "Implement auth module"
        assert sample_plan.rationale == "Breaking auth into implementation and tests"

    def test_plan_empty_tasks(self):
        plan = Plan(tasks=[], rationale="Nothing to do")
        assert plan.tasks == []
        assert plan.rationale == "Nothing to do"


class TestTestResult:
    def test_pass_result(self, sample_test_result_pass):
        assert sample_test_result_pass.passed is True
        assert sample_test_result_pass.failures == []

    def test_fail_result(self, sample_test_result_fail):
        assert sample_test_result_fail.passed is False
        assert len(sample_test_result_fail.failures) == 2


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0  # 0 means unlimited
        assert s.workspace is not None
        assert s.log_level == "INFO"

    def test_custom_values(self):
        s = Settings(provider=Provider.OPENAI, model="gpt-4", max_cycles=3, max_parallel_tasks=2)
        assert s.provider == Provider.OPENAI
        assert s.model == "gpt-4"
        assert s.max_cycles == 3
        assert s.max_parallel_tasks == 2


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

class TestOrchestratorInit:
    def test_init_defaults(self, mock_client):
        orch = Orchestrator(goal="Build a CLI tool", client=mock_client)
        assert orch.goal == "Build a CLI tool"
        assert orch.cycles == 0
        assert orch._tasks == []

    def test_init_with_settings(self):
        s = Settings(max_cycles=5, max_parallel_tasks=3)
        orch = Orchestrator(goal="Test", settings=s)
        assert orch.settings.max_cycles == 5
        assert orch.settings.max_parallel_tasks == 3


class TestIsDone:
    """Test the _is_done() completion detection logic."""

    def test_no_tasks_is_done(self):
        orch = Orchestrator(goal="empty", client=AsyncMock())
        assert orch._is_done() is True

    def test_all_completed_passed_tests_is_done(self):
        orch = Orchestrator(goal="test", client=AsyncMock())
        orch._tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        orch._last_test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orch._is_done() is True

    def test_failed_task_not_done(self):
        orch = Orchestrator(goal="test", client=AsyncMock())
        orch._tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="failed"),
        ]
        assert orch._is_done() is False

    def test_partial_completion_not_done(self):
        orch = Orchestrator(goal="test", client=AsyncMock())
        orch._tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="pending"),
        ]
        assert orch._is_done() is False

    def test_all_completed_tests_failed_not_done(self):
        orch = Orchestrator(goal="test", client=AsyncMock())
        orch._tasks = [
            TaskModel(id="1", description="t1", status="completed"),
        ]
        orch._last_test_result = TestResult(passed=False, summary="bad", failures=["error"])
        assert orch._is_done() is False

    def test_get_tasks_returns_stored(self):
        orch = Orchestrator(goal="test", client=AsyncMock())
        tasks = [TaskModel(id="1", description="t1")]
        orch._tasks = tasks
        assert orch._get_tasks() is tasks


class TestMaxCycles:
    """Test that max_cycles is enforced."""

    @pytest.mark.asyncio
    async def test_max_cycles_stops_loop(self, mock_client):
        settings = Settings(max_cycles=2)
        orch = Orchestrator(goal="test", client=mock_client, settings=settings)
        orch._is_done = MagicMock(return_value=False)  # Always want to continue

        with patch.object(orch, "_cycle", new_callable=AsyncMock) as mock_cycle:
            await orch.run()

        # Should have run exactly 2 cycles then stopped
        assert mock_cycle.call_count == 2
        assert orch.cycles == 2

    @pytest.mark.asyncio
    async def test_unlimited_cycles_when_max_is_zero(self, mock_client):
        settings = Settings(max_cycles=0)
        orch = Orchestrator(goal="test", client=mock_client, settings=settings)
        call_count = 0

        async def _cycle():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                orch._tasks = [TaskModel(id="1", description="t", status="completed")]
                orch._last_test_result = TestResult(passed=True, summary="ok", failures=[])

        with patch.object(orch, "_cycle", side_effect=_cycle):
            with patch.object(orch, "_is_done", side_effect=lambda: call_count >= 3):
                await orch.run()

        assert call_count == 3
        assert orch.cycles == 3


class TestOrchestratorRun:
    """Integration-style tests with mocked subagents."""

    @pytest.mark.asyncio
    async def test_full_cycle_success(self, mock_client):
        """Test a complete cycle: plan, execute, test, done."""
        settings = Settings()
        orch = Orchestrator(goal="Build feature", client=mock_client, settings=settings)

        # Mock the planner to return a plan
        plan = Plan(
            tasks=[TaskModel(id="1", description="Implement feature", files=[], dependencies=[])],
            rationale="Simple task",
        )

        with patch.object(PlannerAgent, "plan", new=AsyncMock(return_value=plan)):
            with patch.object(WorkerAgent, "run", new=AsyncMock(return_value="Implemented feature")):
                with patch.object(TesterAgent, "run", new=AsyncMock(
                    return_value=TestResult(passed=True, summary="All passed", failures=[])
                )):
                    await orch.run()

        assert orch._tasks[0].status == "completed"
        assert orch._last_test_result.passed is True

    @pytest.mark.asyncio
    async def test_full_cycle_with_failures(self, mock_client):
        """Test that failures update the goal for the next cycle."""
        settings = Settings()
        orch = Orchestrator(goal="Build feature", client=mock_client, settings=settings)

        plan = Plan(
            tasks=[TaskModel(id="1", description="Implement feature", files=[], dependencies=[])],
            rationale="Simple task",
        )

        failure_msg = "test_auth.py:12: AssertionError"
        test_result = TestResult(
            passed=False, summary="Tests failed", failures=[failure_msg]
        )

        with patch.object(PlannerAgent, "plan", new=AsyncMock(return_value=plan)):
            with patch.object(WorkerAgent, "run", new=AsyncMock(return_value="Implemented")):
                with patch.object(TesterAgent, "run", new=AsyncMock(return_value=test_result)):
                    with patch.object(Orchestrator, "_is_done", return_value=True):
                        await orch.run()

        # Goal should have been updated to fix failures
        assert "Fix failing tests" in orch.goal
        assert failure_msg in orch.goal

    @pytest.mark.asyncio
    async def test_planning_exception_handled(self, mock_client):
        """Test that planning exceptions don't crash the orchestrator."""
        settings = Settings()
        orch = Orchestrator(goal="test", client=mock_client, settings=settings)

        with patch.object(PlannerAgent, "plan", side_effect=ValueError("LLM error")):
            await orch._cycle()

        assert orch._tasks == []

    @pytest.mark.asyncio
    async def test_task_limit_enforced(self, mock_client):
        """Test that max_parallel_tasks limits the number of tasks."""
        settings = Settings(max_parallel_tasks=2)
        orch = Orchestrator(goal="test", client=mock_client, settings=settings)

        plan = Plan(
            tasks=[
                TaskModel(id=str(i), description=f"Task {i}", files=[], dependencies=[])
                for i in range(1, 6)  # 5 tasks
            ],
            rationale="Many tasks",
        )

        with patch.object(PlannerAgent, "plan", new=AsyncMock(return_value=plan)):
            with patch.object(WorkerAgent, "run", new=AsyncMock(return_value="Done")):
                with patch.object(TesterAgent, "run", new=AsyncMock(
                    return_value=TestResult(passed=True, summary="ok", failures=[])
                )):
                    await orch._cycle()

        assert len(orch._tasks) == 2  # Limited to 2

    @pytest.mark.asyncio
    async def test_event_callback_invoked(self, mock_client):
        """Test that on_event callback receives lifecycle events."""
        received_events: list[str] = []

        async def on_event(event: str, **data: Any) -> None:
            received_events.append(event)

        settings = Settings()
        orch = Orchestrator(
            goal="test", client=mock_client, settings=settings, on_event=on_event
        )

        plan = Plan(
            tasks=[TaskModel(id="1", description="Task", files=[], dependencies=[])],
            rationale="Test",
        )

        with patch.object(PlannerAgent, "plan", new=AsyncMock(return_value=plan)):
            with patch.object(WorkerAgent, "run", new=AsyncMock(return_value="Done")):
                with patch.object(TesterAgent, "run", new=AsyncMock(
                    return_value=TestResult(passed=True, summary="ok", failures=[])
                )):
                    await orch._cycle()

        assert "cycle_start" in received_events or "plan" in received_events
        assert "execute" in received_events
        assert "test" in received_events

    @pytest.mark.asyncio
    async def test_task_failure_marks_status(self, mock_client):
        """Test that worker exceptions mark tasks as failed."""
        settings = Settings()
        orch = Orchestrator(goal="test", client=mock_client, settings=settings)

        plan = Plan(
            tasks=[
                TaskModel(id="1", description="Task 1", files=[], dependencies=[]),
                TaskModel(id="2", description="Task 2", files=[], dependencies=[]),
            ],
            rationale="Test",
        )

        # First worker succeeds, second raises
        mock_run = AsyncMock(side_effect=["Done", RuntimeError("Worker crashed")])

        with patch.object(PlannerAgent, "plan", new=AsyncMock(return_value=plan)):
            with patch.object(WorkerAgent, "run", mock_run):
                with patch.object(TesterAgent, "run", new=AsyncMock(
                    return_value=TestResult(passed=True, summary="ok", failures=[])
                )):
                    await orch._cycle()

        assert orch._tasks[0].status == "completed"
        assert orch._tasks[1].status == "failed"
        assert "crashed" in orch._tasks[1].result
