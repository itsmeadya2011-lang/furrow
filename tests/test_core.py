from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_init():
    orch = Orchestrator(goal="test goal")
    assert orch.goal == "test goal"
    assert orch.plan is None
    assert orch.cycles == 0


def test_orchestrator_is_done_no_plan():
    orch = Orchestrator(goal="test")
    assert orch._is_done() is False


def test_orchestrator_is_done_all_completed():
    orch = Orchestrator(goal="test")
    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    assert orch._is_done() is True


def test_orchestrator_is_done_has_failed():
    orch = Orchestrator(goal="test")
    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="ok",
    )
    assert orch._is_done() is False


def test_orchestrator_is_done_partial():
    orch = Orchestrator(goal="test")
    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ],
        rationale="ok",
    )
    assert orch._is_done() is False


def test_worker_prompt_includes_file_format():
    from furrow.agents.prompts import WORKER_PROMPT
    assert "```" in WORKER_PROMPT
    assert "filename" in WORKER_PROMPT or "file path" in WORKER_PROMPT


@pytest.mark.asyncio
async def test_worker_apply_changes():
    client = MagicMock()
    client.settings.workspace = Path("/tmp/furrow-test-worker")
    client.write_file = AsyncMock()

    worker = WorkerAgent(
        task=TaskModel(id="1", description="do thing", files=["src/main.py"]),
        client=client,
        workspace=Path("/tmp/furrow-test-worker"),
    )

    response = 'Here is the updated file:\n```src/main.py\nprint("hello")\n```\nDone.'
    written = await worker._apply_changes(response)
    assert written == ["src/main.py"]
    client.write_file.assert_called_once()
    args = client.write_file.call_args
    assert args[0][0] == Path("/tmp/furrow-test-worker/src/main.py")
    assert args[0][1] == 'print("hello")'


@pytest.mark.asyncio
async def test_worker_apply_no_changes():
    client = MagicMock()
    client.settings.workspace = Path("/tmp/furrow-test-worker")
    client.write_file = AsyncMock()

    worker = WorkerAgent(
        task=TaskModel(id="1", description="do thing"),
        client=client,
        workspace=Path("/tmp/furrow-test-worker"),
    )

    written = await worker._apply_changes("Just a summary, no code blocks.")
    assert written == []
    client.write_file.assert_not_called()


@pytest.mark.asyncio
async def test_tester_uses_workspace():
    client = MagicMock()
    client.settings.workspace = Path("/tmp/furrow-test-tester")
    client.complete = AsyncMock(return_value=json.dumps({"passed": True, "summary": "ok", "failures": []}))

    tester = TesterAgent(client=client, workspace=Path("/tmp/furrow-test-tester"))

    # Mock the subprocess to avoid actually running tests
    import asyncio
    original_create_subprocess_exec = asyncio.create_subprocess_exec

    async def mock_create_subprocess_exec(*cmd, **kwargs):
        # Verify cwd is set to workspace
        assert kwargs.get("cwd") == str(Path("/tmp/furrow-test-tester"))
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        return mock_proc

    asyncio.create_subprocess_exec = mock_create_subprocess_exec
    try:
        result = await tester.run("test goal", [])
        assert result.passed is True
    finally:
        asyncio.create_subprocess_exec = original_create_subprocess_exec


@pytest.mark.asyncio
async def test_orchestrator_on_status_callback():
    events: list[tuple[str, dict]] = []

    async def callback(event: str, payload: dict) -> None:
        events.append((event, payload))

    orch = Orchestrator(goal="test", on_status=callback)
    await orch._notify("test_event", {"key": "value"})
    assert len(events) == 1
    assert events[0] == ("test_event", {"key": "value"})


@pytest.mark.asyncio
async def test_orchestrator_on_status_callback_noop_on_exception():
    calls = 0

    async def bad_callback(event: str, payload: dict) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    orch = Orchestrator(goal="test", on_status=bad_callback)
    await orch._notify("event", {})
    await orch._notify("event2", {})
    assert calls == 2  # Should not crash orchestrator


def test_tester_default_workspace_from_settings():
    client = MagicMock()
    client.settings.workspace = Path("/default/workspace")
    tester = TesterAgent(client=client)
    assert tester.workspace == Path("/default/workspace")


def test_worker_default_workspace_from_settings():
    client = MagicMock()
    client.settings.workspace = Path("/default/workspace")
    task = TaskModel(id="1", description="test")
    worker = WorkerAgent(task=task, client=client)
    assert worker.workspace == Path("/default/workspace")
