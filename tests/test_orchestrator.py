from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


@pytest.fixture
def settings() -> Settings:
    return Settings(provider=Provider.ANTHROPIC, model="test-model")


@pytest.fixture
def mock_client(settings: Settings) -> LLMClient:
    client = LLMClient(settings=settings)
    client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')  # type: ignore[method-assign]
    return client


def _plan_with_tasks() -> Plan:
    return Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b"),
        ],
        rationale="ok",
    )


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit_cycle(self, cycle: int) -> None:
        self.events.append({"type": "cycle", "cycle": cycle})

    def emit_plan(self, plan) -> None:
        self.events.append({"type": "plan", "num_tasks": len(plan.tasks)})

    def emit_task(self, task_id: str, status: str, result=None) -> None:
        self.events.append({"type": "task", "id": task_id, "status": status})

    def emit_tests(self, test_result) -> None:
        self.events.append({"type": "tests", "passed": test_result.passed})

    def emit_done(self, reason: str) -> None:
        self.events.append({"type": "done", "reason": reason})

    def emit_error(self, message: str) -> None:
        self.events.append({"type": "error", "message": message})


class TestOrchestratorGetTasks:
    def test_get_tasks_before_cycle(self) -> None:
        orch = Orchestrator(goal="x")
        assert orch._get_tasks() == []

    def test_get_tasks_after_cycle(self) -> None:
        orch = Orchestrator(goal="x")
        orch.current_plan = _plan_with_tasks()
        tasks = orch._get_tasks()
        assert len(tasks) == 2
        assert tasks[0].description == "a"


class TestOrchestratorRun:
    @pytest.mark.asyncio
    async def test_max_cycles_caps_run(self, settings: Settings) -> None:
        # Worker always fails → _is_done() never True → only max_cycles can halt.
        orch = Orchestrator(goal="build x", max_cycles=2, client=LLMClient(settings=settings))
        plan = _plan_with_tasks()

        with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
             patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockPlanner.return_value.plan = AsyncMock(return_value=plan)
            MockWorker.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))
            MockTester.return_value.run = AsyncMock(
                return_value=TestResult(passed=False, summary="bad", failures=["err"])
            )
            await orch.run()

        # max_cycles=2 → runs exactly 2 cycles, then halts.
        assert MockPlanner.return_value.plan.await_count == 2

    @pytest.mark.asyncio
    async def test_max_cycles_one_runs_one_cycle(self, settings: Settings) -> None:
        orch = Orchestrator(goal="build x", max_cycles=1, client=LLMClient(settings=settings))
        plan = _plan_with_tasks()

        with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
             patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockPlanner.return_value.plan = AsyncMock(return_value=plan)
            MockWorker.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))
            MockTester.return_value.run = AsyncMock(
                return_value=TestResult(passed=False, summary="bad", failures=["err"])
            )
            await orch.run()

        # max_cycles=1 → exactly 1 cycle.
        assert MockPlanner.return_value.plan.await_count == 1

    @pytest.mark.asyncio
    async def test_completion_when_no_more_tasks(self, settings: Settings) -> None:
        orch = Orchestrator(goal="build x", client=LLMClient(settings=settings))

        with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockPlanner.return_value.plan = AsyncMock(
                return_value=Plan(tasks=[], rationale="none")
            )
            MockTester.return_value.run = AsyncMock()
            await orch.run()

        MockPlanner.return_value.plan.assert_awaited_once()
        # Tester should not run because _cycle() returns early when plan.tasks is empty
        MockTester.return_value.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_event_bus_receives_events(self, settings: Settings) -> None:
        orch = Orchestrator(
            goal="build x",
            client=LLMClient(settings=settings),
            event_bus=_FakeBus(),
        )
        plan = _plan_with_tasks()
        test_passed = TestResult(passed=True, summary="ok", failures=[])

        with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
             patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockPlanner.return_value.plan = AsyncMock(return_value=plan)
            MockWorker.return_value.run = AsyncMock(return_value="done")
            MockTester.return_value.run = AsyncMock(return_value=test_passed)
            await orch.run()

        event_types = {e["type"] for e in orch.event_bus.events}
        assert "cycle" in event_types
        assert "plan" in event_types
        assert "task" in event_types
        assert "tests" in event_types
        assert "done" in event_types