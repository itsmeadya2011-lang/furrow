from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakePlanner:
    """Fake planner that returns a pre-built plan."""
    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    async def plan(self, goal: str) -> Plan:
        return self._plan


class FakeWorker:
    """Fake worker that returns a fixed string."""
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self) -> str:
        self.calls.append("called")
        return "done"


class FakeTester:
    """Fake tester that returns a fixed test result."""
    def __init__(self, result: TestResult) -> None:
        self._result = result

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return self._result


@pytest.fixture
def completed_plan() -> Plan:
    return Plan(
        tasks=[
            TaskModel(id="1", description="task one"),
            TaskModel(id="2", description="task two"),
        ],
        rationale="test plan",
    )


@pytest.fixture
def failed_plan() -> Plan:
    return Plan(
        tasks=[
            TaskModel(id="1", description="task one", status="failed", result="boom"),
        ],
        rationale="test plan",
    )


def _make_orchestrator(goal: str = "test goal", plan: Plan | None = None, test_result: TestResult | None = None) -> Orchestrator:
    orch = Orchestrator(goal=goal)
    orch.planner = FakePlanner(plan or Plan(tasks=[], rationale="empty"))
    return orch


@pytest.mark.asyncio
async def test_tasks_persisted_after_cycle(completed_plan: Plan):
    """_get_tasks() should return plan tasks after _cycle() runs."""
    orch = _make_orchestrator(plan=completed_plan)
    with patch("furrow.core.orchestrator.WorkerAgent") as W, patch("furrow.core.orchestrator.TesterAgent") as T:
        W.return_value.run = AsyncMock(return_value="ok")
        T.return_value.run = AsyncMock(return_value=TestResult(passed=True, summary="ok"))

        await orch._cycle()

    tasks = orch._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"
    assert tasks[0].status == "completed"
    assert tasks[1].status == "completed"


@pytest.mark.asyncio
async def test_is_done_all_completed(completed_plan: Plan):
    """_is_done() should return True when all tasks are completed."""
    orch = _make_orchestrator(plan=completed_plan)
    with patch("furrow.core.orchestrator.WorkerAgent") as W, patch("furrow.core.orchestrator.TesterAgent") as T:
        W.return_value.run = AsyncMock(return_value="ok")
        T.return_value.run = AsyncMock(return_value=TestResult(passed=True, summary="ok"))

        await orch._cycle()

    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_not_done_with_failures(failed_plan: Plan):
    """_is_done() should return False when any task failed."""
    orch = _make_orchestrator(plan=Plan(tasks=[], rationale="empty"))
    # Simulate persisted tasks with a failed status (as would happen after _cycle)
    orch._tasks = [TaskModel(id="1", description="t", status="failed", result="boom")]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_empty_tasks():
    """_is_done() should return False when there are no tasks."""
    orch = _make_orchestrator(plan=Plan(tasks=[], rationale="empty"))
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_max_cycles_respected():
    """Orchestrator should stop after max_cycles even if goal not complete."""
    orch = _make_orchestrator(
        plan=Plan(tasks=[TaskModel(id="1", description="t1")], rationale="ok")
    )
    orch.settings.max_cycles = 2
    orch.settings.max_parallel_tasks = 5

    with patch("furrow.core.orchestrator.WorkerAgent") as W, patch("furrow.core.orchestrator.TesterAgent") as T:
        # Worker raises so task status = "failed", _is_done() returns False,
        # and the loop continues until max_cycles is hit.
        W.return_value.run = AsyncMock(side_effect=Exception("fail"))
        T.return_value.run = AsyncMock(return_value=TestResult(passed=True, summary="ok"))

        await orch.run()

    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_on_event_callback():
    """on_event callback should receive status messages."""
    events: list[str] = []

    async def capture(msg: str) -> None:
        events.append(msg)

    orch = Orchestrator(
        goal="test",
        on_event=capture,
    )
    orch.planner = FakePlanner(Plan(tasks=[TaskModel(id="1", description="t1")], rationale="ok"))
    orch.settings.max_cycles = 0  # infinite; loop stops via _is_done()

    with patch("furrow.core.orchestrator.WorkerAgent") as W, patch("furrow.core.orchestrator.TesterAgent") as T:
        W.return_value.run = AsyncMock(return_value="ok")
        T.return_value.run = AsyncMock(return_value=TestResult(passed=True, summary="ok"))

        await orch.run()

    assert len(events) > 0
    assert any("Starting Furrow" in e for e in events)
    assert any("Cycle 1" in e for e in events)
    assert any("Goal complete" in e for e in events)


@pytest.mark.asyncio
async def test_get_state():
    """get_state() should return serializable state."""
    orch = _make_orchestrator(plan=Plan(tasks=[TaskModel(id="1", description="t1")], rationale="ok"))
    state = orch.get_state()
    assert state["goal"] == "test goal"
    assert state["cycles"] == 0
    assert state["tasks"] == []


# Keep original tests
def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
