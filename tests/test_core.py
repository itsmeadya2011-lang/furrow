"""Comprehensive test suite for the Furrow autonomous coding agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

# ─── Config / Model Tests ───────────────────────────────────────────


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_all_fields():
    t = TaskModel(
        id="2",
        description="implement feature",
        files=["src/a.py", "src/b.py"],
        dependencies=["1"],
        status="completed",
        result="done",
    )
    assert t.files == ["src/a.py", "src/b.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"
    assert t.result == "done"


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"


def test_plan_with_dependencies():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
        ],
        rationale="multi-step plan",
    )
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]


def test_test_result_defaults():
    t = TestResult(passed=True, summary="all good")
    assert t.passed is True
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="tests failed", failures=["test_foo failed"])
    assert t.passed is False
    assert len(t.failures) == 1
    assert t.failures[0] == "test_foo failed"


# ─── PlannerAgent Tests ──────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    client = MagicMock(spec=LLMClient)
    client.settings = MagicMock()
    client.settings.planner_model = "test-model"
    client.complete = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_planner_parse_valid_json(mock_llm):
    mock_llm.complete.return_value = '{"tasks": [{"id": "1", "description": "do it", "files": ["a.py"], "dependencies": []}], "rationale": "plan"}'
    planner = PlannerAgent(client=mock_llm)
    plan = await planner.plan("Implement feature X")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "do it"
    assert plan.rationale == "plan"
    mock_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_planner_raises_on_invalid_json(mock_llm):
    mock_llm.complete.return_value = "not valid json at all"
    planner = PlannerAgent(client=mock_llm)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("some goal")


@pytest.mark.asyncio
async def test_planner_with_empty_tasks(mock_llm):
    mock_llm.complete.return_value = '{"tasks": [], "rationale": "goal already met"}'
    planner = PlannerAgent(client=mock_llm)
    plan = await planner.plan("already done")
    assert plan.tasks == []
    assert plan.rationale == "goal already met"


# ─── WorkerAgent Tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_returns_llm_response():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = MagicMock()
    mock_client.settings.worker_model = "worker-model"
    mock_client.complete = AsyncMock(return_value="Implemented the feature")
    task = TaskModel(id="1", description="Write a function")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()
    assert result == "Implemented the feature"
    mock_client.complete.assert_called_once()


# ─── TesterAgent Tests ───────────────────────────────────────────────


@pytest.fixture
def mock_tester_llm():
    client = MagicMock(spec=LLMClient)
    client.settings = MagicMock()
    client.settings.tester_model = "tester-model"
    client.complete = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_tester_passes_json(mock_tester_llm):
    mock_tester_llm.complete.return_value = '{"passed": true, "summary": "all tests passed", "failures": []}'
    tester = TesterAgent(client=mock_tester_llm)
    tester._run_tests = AsyncMock(return_value="")
    result = await tester.run("some goal", [])
    assert result.passed is True
    assert result.summary == "all tests passed"


@pytest.mark.asyncio
async def test_tester_fails_json(mock_tester_llm):
    mock_tester_llm.complete.return_value = '{"passed": false, "summary": "1 test failed", "failures": ["test_foo"]}'
    tester = TesterAgent(client=mock_tester_llm)
    tester._run_tests = AsyncMock(return_value="")
    result = await tester.run("some goal", [])
    assert result.passed is False
    assert len(result.failures) == 1


@pytest.mark.asyncio
async def test_tester_handles_markdown_json(mock_tester_llm):
    """Tester should strip markdown code fences before parsing JSON."""
    mock_tester_llm.complete.return_value = '```json\n{"passed": true, "summary": "ok", "failures": []}\n```'
    tester = TesterAgent(client=mock_tester_llm)
    tester._run_tests = AsyncMock(return_value="")
    result = await tester.run("some goal", [])
    assert result.passed is True
    assert result.summary == "ok"


@pytest.mark.asyncio
async def test_tester_fallback_when_no_json(mock_tester_llm):
    """When JSON parsing fails, fall back to substring matching."""
    mock_tester_llm.complete.return_value = "All tests passed successfully. No failures."
    tester = TesterAgent(client=mock_tester_llm)
    result = tester._parse_tester_response("All tests passed successfully. No failures.")
    assert result.passed is False  # No "passed" substring in lowercase


@patch("furrow.agents.tester.TesterAgent._run_tests", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_tester_handles_exception(mock_run_tests, mock_tester_llm):
    mock_run_tests.side_effect = Exception("test runner crashed")
    tester = TesterAgent(client=mock_tester_llm)
    result = await tester.run("some goal", [])
    assert result.passed is False
    assert "test runner crashed" in result.summary


# ─── Orchestrator Tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_stores_tasks_and_tracks_completion():
    """The orchestrator must store tasks on the instance so _is_done works."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = MagicMock()
    mock_client.settings.max_cycles = 0
    mock_client.settings.max_parallel_tasks = 5

    planner = MagicMock(spec=PlannerAgent)
    planner.plan = AsyncMock(return_value=Plan(
        tasks=[TaskModel(id="1", description="task 1")],
        rationale="test"
    ))

    with patch("furrow.core.orchestrator.PlannerAgent", return_value=planner), patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as mock_worker_cls, patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls:
        mock_worker_instance = MagicMock()
        mock_worker_instance.run = AsyncMock(return_value="done")
        mock_worker_cls.return_value = mock_worker_instance

        mock_tester_instance = MagicMock()
        mock_tester_instance.run = AsyncMock(return_value=TestResult(passed=True, summary="ok"))
        mock_tester_cls.return_value = mock_tester_instance

        events = []

        async def capture(event, data):
            events.append((event, data))

        orch = Orchestrator(
            goal="test goal",
            client=mock_client,
            max_cycles=1,
            on_progress=capture,
        )
        await orch.run()

        # Tasks should be stored on the instance
        assert len(orch._tasks) == 1
        assert orch._tasks[0].status == "completed"
        assert orch._is_done() is True

        # Progress events should have been emitted
        assert any(e == "start" for e, _ in events)
        assert any(e == "cycle_start" for e, _ in events)
        assert any(e == "complete" for e, _ in events)


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles():
    """The orchestrator must stop after max_cycles even if not done."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = MagicMock()
    mock_client.settings.max_cycles = 0
    mock_client.settings.max_parallel_tasks = 5

    planner = MagicMock(spec=PlannerAgent)
    planner.plan = AsyncMock(return_value=Plan(
        tasks=[TaskModel(id="1", description="task 1")],
        rationale="test"
    ))

    with patch("furrow.core.orchestrator.PlannerAgent", return_value=planner), patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as mock_worker_cls, patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls:
        mock_worker_instance = MagicMock()
        mock_worker_instance.run = AsyncMock(return_value="done")
        mock_worker_cls.return_value = mock_worker_instance

        mock_tester_instance = MagicMock()
        mock_tester_instance.run = AsyncMock(return_value=TestResult(passed=False, summary="fail"))
        mock_tester_cls.return_value = mock_tester_instance

        orch = Orchestrator(
            goal="test goal",
            client=mock_client,
            max_cycles=3,
        )
        await orch.run()

        assert orch.cycles == 3


@pytest.mark.asyncio
async def test_orchestrator_failed_tasks_not_done():
    """When tasks fail, _is_done should return False."""
    orch = Orchestrator(goal="test", max_cycles=0)
    orch._tasks = [
        TaskModel(id="1", description="task 1", status="failed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_all_completed_is_done():
    """When all tasks complete, _is_done should return True."""
    orch = Orchestrator(goal="test", max_cycles=0)
    orch._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_no_tasks_means_done():
    """When no tasks are planned, the orchestrator should stop."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = MagicMock()
    mock_client.settings.max_cycles = 0
    mock_client.settings.max_parallel_tasks = 5

    planner = MagicMock(spec=PlannerAgent)
    planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="nothing to do"))

    with patch("furrow.core.orchestrator.PlannerAgent", return_value=planner):
        orch = Orchestrator(goal="test goal", client=mock_client, max_cycles=1)
        await orch.run()
        assert orch.cycles == 1
