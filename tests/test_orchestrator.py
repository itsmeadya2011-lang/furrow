from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _settings(**kwargs) -> Settings:
    data = {"provider": Provider.ANTHROPIC, "anthropic_api_key": "sk-ant-test", **kwargs}
    return Settings(**data)


def _make_task(id_: str, status: str = "pending") -> TaskModel:
    t = TaskModel(id=id_, description=f"task {id_}")
    t.status = status
    return t


def _make_plan(*tasks: TaskModel) -> Plan:
    return Plan(tasks=list(tasks), rationale="plan")


# --- _is_done ---


def test_is_done_no_tasks():
    settings = _settings()
    orch = Orchestrator(goal="x", config=settings)
    assert orch._is_done() is True


def test_is_done_all_completed():
    settings = _settings()
    orch = Orchestrator(goal="x", config=settings)
    orch.last_plan = _make_plan(_make_task("1", "completed"), _make_task("2", "completed"))
    assert orch._is_done() is True


def test_is_done_all_failed():
    settings = _settings()
    orch = Orchestrator(goal="x", config=settings)
    orch.last_plan = _make_plan(_make_task("1", "failed"), _make_task("2", "failed"))
    assert orch._is_done() is True


def test_is_done_mixed_continues():
    settings = _settings()
    orch = Orchestrator(goal="x", config=settings)
    orch.last_plan = _make_plan(_make_task("1", "completed"), _make_task("2", "failed"))
    assert orch._is_done() is False


def test_is_done_some_completed():
    settings = _settings()
    orch = Orchestrator(goal="x", config=settings)
    orch.last_plan = _make_plan(_make_task("1", "completed"))
    assert orch._is_done() is True


# --- run() cycle control ---


@pytest.mark.asyncio
async def test_run_stops_when_all_completed():
    settings = _settings(max_cycles=10)
    orch = Orchestrator(goal="x", config=settings)

    plan = _make_plan(_make_task("1", "completed"))

    mock_planner = AsyncMock(return_value=plan)
    mock_worker = AsyncMock(return_value="done")
    mock_tester = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )

    with patch.object(orch, "planner", mock_planner):
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                await orch.run()

    assert orch.cycles == 1
    mock_planner.plan.assert_called_once_with("x")


@pytest.mark.asyncio
async def test_run_stops_at_max_cycles():
    settings = _settings(max_cycles=2)
    orch = Orchestrator(goal="x", config=settings)

    # Plan with one pending task every time, tests pass
    plan = _make_plan(_make_task("1", "pending"))

    mock_planner = AsyncMock(return_value=plan)
    mock_worker = AsyncMock(return_value="done")
    mock_tester = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )

    with patch.object(orch, "planner", mock_planner):
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                await orch.run()

    assert orch.cycles == 2
    assert mock_planner.plan.call_count == 2


@pytest.mark.asyncio
async def test_run_continues_on_failure():
    settings = _settings(max_cycles=10)
    orch = Orchestrator(goal="x", config=settings)

    plan = _make_plan(_make_task("1", "pending"))

    mock_planner = AsyncMock(return_value=plan)
    mock_worker = AsyncMock(return_value="done")
    mock_tester = AsyncMock(
        return_value=TestResult(passed=False, summary="fail", failures=["error"])
    )

    with patch.object(orch, "planner", mock_planner):
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                await orch.run()

    # First cycle: tasks complete, tests fail -> goal updated, continue
    # Second cycle: tasks complete, tests fail -> goal updated, continue...
    # It will run until max_cycles (10)
    assert orch.cycles == 10
    # Goal should be updated with fix instructions
    assert "Fix failing tests:" in orch.goal


@pytest.mark.asyncio
async def test_run_progress_callback():
    settings = _settings(max_cycles=1)
    orch = Orchestrator(goal="x", config=settings)

    plan = _make_plan(_make_task("1", "completed"))

    mock_planner = AsyncMock(return_value=plan)
    mock_worker = AsyncMock(return_value="done")
    mock_tester = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )

    progress_messages = []

    async def on_progress(msg: str) -> None:
        progress_messages.append(msg)

    orch.on_progress = on_progress

    with patch.object(orch, "planner", mock_planner):
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                await orch.run()

    assert "Cycle 1 starting" in progress_messages
    assert "Planning..." in progress_messages
    assert "Task 1 completed" in progress_messages
