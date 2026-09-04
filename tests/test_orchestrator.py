from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator, _noop_event
from furrow.llm import LLMClient


@pytest.fixture
def mock_client() -> LLMClient:
    client = LLMClient()
    client.complete = AsyncMock(return_value='{"tasks": [{"id": "1", "description": "work", "files": [], "dependencies": [], "status": "pending", "result": null}], "rationale": "test"}')
    return client


@pytest.fixture
def orchestrator(mock_client: LLMClient) -> Orchestrator:
    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        mock_planner = AsyncMock()
        mock_planner.plan = AsyncMock(return_value=Plan(tasks=[TaskModel(id="1", description="work")], rationale="test"))
        MockPlanner.return_value = mock_planner
        mock_tester = AsyncMock()
        mock_tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))
        MockTester.return_value = mock_tester
        return Orchestrator(goal="test goal", client=mock_client)


def test_orchestrator_init(orchestrator: Orchestrator):
    assert orchestrator.goal == "test goal"
    assert orchestrator.cycles == 0
    assert orchestrator.current_plan is None
    assert orchestrator.on_event is _noop_event


def test_orchestrator_init_with_callback():
    events = []
    def callback(event_type: str, data: dict) -> None:
        events.append((event_type, data))
    orchestrator = Orchestrator(goal="test", on_event=callback)
    assert orchestrator.on_event is callback
    orchestrator._emit("test_event", {"key": "value"})
    assert events == [("test_event", {"key": "value"})]


def test_orchestrator_get_tasks_no_plan(orchestrator: Orchestrator):
    assert orchestrator._get_tasks() == []


def test_orchestrator_is_done_no_tasks(orchestrator: Orchestrator):
    orchestrator.current_plan = Plan(tasks=[], rationale="none")
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_all_completed(orchestrator: Orchestrator):
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_some_failed(orchestrator: Orchestrator):
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed", result="error"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_incomplete(orchestrator: Orchestrator):
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_cycle_stores_plan(orchestrator: Orchestrator, mock_client: LLMClient):
    mock_plan = Plan(tasks=[TaskModel(id="1", description="do work")], rationale="test")
    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner:
        mock_planner = AsyncMock()
        mock_planner.plan = AsyncMock(return_value=mock_plan)
        MockPlanner.return_value = mock_planner
        await orchestrator._cycle()
    assert orchestrator.current_plan is mock_plan
    assert orchestrator.current_plan.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_orchestrator_cycle_no_tasks(orchestrator: Orchestrator, mock_client: LLMClient):
    mock_plan = Plan(tasks=[], rationale="none")
    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner:
        mock_planner = AsyncMock()
        mock_planner.plan = AsyncMock(return_value=mock_plan)
        MockPlanner.return_value = mock_planner
        await orchestrator._cycle()
    assert orchestrator.current_plan is mock_plan


@pytest.mark.asyncio
async def test_orchestrator_cycle_emits_events(orchestrator: Orchestrator, mock_client: LLMClient):
    events = []
    def on_event(event_type: str, data: dict) -> None:
        events.append((event_type, data))
    orchestrator.on_event = on_event

    mock_plan = Plan(tasks=[TaskModel(id="1", description="do work")], rationale="test")
    mock_test = TestResult(passed=True, summary="ok", failures=[])
    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        mock_planner = AsyncMock()
        mock_planner.plan = AsyncMock(return_value=mock_plan)
        MockPlanner.return_value = mock_planner
        mock_tester = AsyncMock()
        mock_tester.run = AsyncMock(return_value=mock_test)
        MockTester.return_value = mock_tester
        await orchestrator._cycle()
    event_types = [e[0] for e in events]
    assert "plan_generated" in event_types
    assert "task_started" in event_types
    assert "task_completed" in event_types
    assert "tests_passed" in event_types


@pytest.mark.asyncio
async def test_orchestrator_run_stops_when_done(orchestrator: Orchestrator, mock_client: LLMClient):
    mock_plan = Plan(tasks=[TaskModel(id="1", description="do work")], rationale="test")
    mock_test = TestResult(passed=True, summary="ok", failures=[])
    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        mock_planner = AsyncMock()
        mock_planner.plan = AsyncMock(return_value=mock_plan)
        MockPlanner.return_value = mock_planner
        mock_tester = AsyncMock()
        mock_tester.run = AsyncMock(return_value=mock_test)
        MockTester.return_value = mock_tester
        await orchestrator.run()
    assert orchestrator.cycles == 1
