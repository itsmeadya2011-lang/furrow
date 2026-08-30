import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import RetryError

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# --- Config / model tests ---

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# --- Orchestrator tests ---

def test_orchestrator_done_no_tasks():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.tasks = []
    orchestrator.cycles = 0
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 0
    assert orchestrator._is_done() is True


def test_orchestrator_done_all_completed():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orchestrator.cycles = 1
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 0
    assert orchestrator._is_done() is True


def test_orchestrator_not_done_failed():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.tasks = [
        TaskModel(id="1", description="a", status="failed"),
    ]
    orchestrator.cycles = 1
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 0
    assert orchestrator._is_done() is False


def test_orchestrator_not_done_pending():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.tasks = [
        TaskModel(id="1", description="a", status="pending"),
    ]
    orchestrator.cycles = 1
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 0
    assert orchestrator._is_done() is False


def test_orchestrator_max_cycles_enforced():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.tasks = [
        TaskModel(id="1", description="a", status="completed"),
    ]
    orchestrator.cycles = 5
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 5
    assert orchestrator._is_done() is True


def test_orchestrator_get_tasks():
    orchestrator = Orchestrator.__new__(Orchestrator)
    tasks = [TaskModel(id="1", description="a")]
    orchestrator.tasks = tasks
    assert orchestrator._get_tasks() is tasks


@pytest.mark.asyncio
async def test_orchestrator_cycle_success():
    mock_client = MagicMock()
    mock_client.settings = Settings(provider=Provider.ANTHROPIC, max_cycles=0)

    mock_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=mock_plan)

    mock_worker = MagicMock()
    mock_worker.run = AsyncMock(return_value="done")

    mock_tester = MagicMock()
    mock_tester.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )

    events = []

    async def capture_event(event, data):
        events.append((event, data))

    with patch("furrow.core.orchestrator.PlannerAgent", return_value=mock_planner):
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                orchestrator = Orchestrator(
                    goal="test", client=mock_client, on_event=capture_event
                )
                await orchestrator._cycle()

    assert orchestrator.tasks[0].status == "completed"
    assert orchestrator.tasks[0].result == "done"
    assert len(events) == 3
    assert events[0][0] == "plan"
    assert events[1][0] == "task_update"
    assert events[2][0] == "test_result"


@pytest.mark.asyncio
async def test_orchestrator_cycle_failure_updates_goal():
    mock_client = MagicMock()
    mock_client.settings = Settings(provider=Provider.ANTHROPIC, max_cycles=0)

    mock_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=mock_plan)

    mock_worker = MagicMock()
    mock_worker.run = AsyncMock(return_value="done")

    mock_tester = MagicMock()
    mock_tester.run = AsyncMock(
        return_value=TestResult(passed=False, summary="fail", failures=["error1"])
    )

    with patch("furrow.core.orchestrator.PlannerAgent", return_value=mock_planner):
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                orchestrator = Orchestrator(goal="test", client=mock_client)
                await orchestrator._cycle()

    assert "Fix failing tests" in orchestrator.goal
    assert "error1" in orchestrator.goal


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles():
    mock_client = MagicMock()
    mock_client.settings = Settings(provider=Provider.ANTHROPIC, max_cycles=2)

    mock_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=mock_plan)

    mock_worker = MagicMock()
    mock_worker.run = AsyncMock(side_effect=Exception("worker error"))

    mock_tester = MagicMock()
    mock_tester.run = AsyncMock(
        return_value=TestResult(passed=False, summary="fail", failures=["error1"])
    )

    with patch("furrow.core.orchestrator.PlannerAgent", return_value=mock_planner):
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                orchestrator = Orchestrator(goal="test", client=mock_client)
                await orchestrator.run()

    assert orchestrator.cycles == 2


# --- LLM client tests ---

def test_llm_client_missing_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient.__new__(LLMClient)
    client.settings = Settings(provider=Provider.ANTHROPIC)
    client._anthropic = None
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set.*Configured provider: anthropic"):
        _ = client.anthropic


def test_llm_client_missing_openai_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient.__new__(LLMClient)
    client.settings = Settings(provider=Provider.OPENAI)
    client._openai = None
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set.*Configured provider: openai"):
        _ = client.openai


# --- Planner tests ---

@pytest.mark.asyncio
async def test_planner_raises_on_invalid_json():
    client = MagicMock()
    client.complete = AsyncMock(return_value="not json")
    client.settings = Settings(provider=Provider.ANTHROPIC)
    planner = PlannerAgent(client=client)
    with pytest.raises(RetryError):
        await planner.plan("do something")


@pytest.mark.asyncio
async def test_planner_succeeds_with_valid_json():
    client = MagicMock()
    client.complete = AsyncMock(
        return_value='{"tasks": [{"id": "1", "description": "x", "files": [], "dependencies": []}], "rationale": "ok"}'
    )
    client.settings = Settings(provider=Provider.ANTHROPIC)
    planner = PlannerAgent(client=client)
    plan = await planner.plan("do something")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"


# --- Tester tests ---

@pytest.mark.asyncio
async def test_tester_returns_first_successful_runner():
    client = MagicMock()
    client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "ok", "failures": []}'
    )
    client.settings = Settings(provider=Provider.ANTHROPIC)
    tester = TesterAgent(client=client)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"tests passed\n", b""))

    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        output = await tester._run_tests()
        assert "tests passed" in output


@pytest.mark.asyncio
async def test_tester_returns_last_failed_output():
    client = MagicMock()
    client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "ok", "failures": []}'
    )
    client.settings = Settings(provider=Provider.ANTHROPIC)
    tester = TesterAgent(client=client)

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"tests failed\n", b""))

    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        output = await tester._run_tests()
        assert "tests failed" in output


@pytest.mark.asyncio
async def test_tester_no_runner_found():
    client = MagicMock()
    client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "ok", "failures": []}'
    )
    client.settings = Settings(provider=Provider.ANTHROPIC)
    tester = TesterAgent(client=client)

    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError,
    ):
        output = await tester._run_tests()
        assert "No test runner found" in output
