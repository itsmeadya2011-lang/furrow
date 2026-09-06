import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_completed_tasks():
    client = AsyncMock(spec=LLMClient)
    client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator._last_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_pending_tasks():
    client = AsyncMock(spec=LLMClient)
    client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator._last_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_failed_tasks():
    client = AsyncMock(spec=LLMClient)
    client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator._last_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_empty_plan():
    client = AsyncMock(spec=LLMClient)
    client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator._last_plan = Plan(tasks=[], rationale="ok")
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_max_cycles_halts():
    client = AsyncMock(spec=LLMClient)
    client.settings.max_cycles = 1
    planner = AsyncMock()
    planner.plan.return_value = Plan(tasks=[], rationale="done")
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.planner = planner
    await orchestrator.run()
    assert orchestrator.cycles == 1


@pytest.mark.asyncio
async def test_planner_json_recovery_from_markdown():
    client = AsyncMock(spec=LLMClient)
    client.complete.return_value = '```json\n{"tasks": [{"id": "1", "description": "x", "files": [], "dependencies": []}], "rationale": "ok"}\n```'
    client.settings.planner_model = "test"
    planner = PlannerAgent(client=client)
    plan = await planner.plan("do something")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "x"


@pytest.mark.asyncio
async def test_planner_json_recovery_plain():
    client = AsyncMock(spec=LLMClient)
    client.complete.return_value = '{"tasks": [{"id": "1", "description": "x", "files": [], "dependencies": []}], "rationale": "ok"}'
    client.settings.planner_model = "test"
    planner = PlannerAgent(client=client)
    plan = await planner.plan("do something")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "x"


@pytest.mark.asyncio
async def test_tester_uses_workspace():
    client = AsyncMock(spec=LLMClient)
    client.settings.workspace = "/tmp/fake_workspace"
    client.complete.return_value = '{"passed": true, "summary": "ok", "failures": []}'
    client.settings.tester_model = "test"
    tester = TesterAgent(client=client)

    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"ok", b"")

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess:
        mock_subprocess.return_value = mock_proc
        result = await tester.run("goal", [])
        mock_subprocess.assert_called_once()
        _, kwargs = mock_subprocess.call_args
        assert kwargs.get("cwd") == "/tmp/fake_workspace"
        assert result.passed is True


def test_queue_file_writes():
    from furrow.web.server import _QueueFile
    queue: asyncio.Queue[str] = asyncio.Queue()
    qf = _QueueFile(queue)
    qf.write("hello")
    assert queue.get_nowait() == "hello"


