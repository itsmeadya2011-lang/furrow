from __future__ import annotations

from unittest.mock import AsyncMock, patch

from furrow.config import TestResult
from furrow.core.orchestrator import Orchestrator


class _FakeSettings:
    planner_model = "planner-model"
    worker_model = "worker-model"
    tester_model = "tester-model"
    max_parallel_tasks = 2
    max_cycles = 2


class _FakeClient:
    def __init__(self, plan=None, worker="done", test=None):
        self.settings = _FakeSettings()
        self._plan = plan
        self._worker = worker
        self._test = test

    async def complete(self, prompt, system="", model=None):
        if "Test output:" in prompt:
            return self._test or (
                '{"passed": true, "summary": "ok", "failures": []}'
            )
        if "Goal:" in prompt and "Task:" not in prompt:
            return self._plan or (
                '{"tasks": [{"id": "1", "description": "d", '
                '"files": [], "dependencies": []}], "rationale": "r"}'
            )
        return self._worker


async def test_run_respects_max_cycles():
    client = _FakeClient(
        test='{"passed": false, "summary": "fail", "failures": ["fail"]}'
    )
    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, patch(
        "furrow.core.orchestrator.TesterAgent"
    ) as MockTester:
        async def bad_run():
            raise RuntimeError("fail")

        MockWorker.return_value.run = bad_run
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["fail"])
        )
        orchestrator = Orchestrator(goal="test", client=client)
        events = []

        def on_event(event):
            events.append(event)

        orchestrator.on_event = on_event
        await orchestrator.run()

    assert orchestrator.cycles == 2
    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["reason"] == "cycle limit reached"


async def test_events_emitted():
    client = _FakeClient(test='{"passed": true, "summary": "ok", "failures": []}')
    with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        orchestrator = Orchestrator(goal="test", client=client)
        events = []

        def on_event(event):
            events.append(event)

        orchestrator.on_event = on_event
        await orchestrator.run()

    event_types = {e["type"] for e in events}
    assert "cycle" in event_types
    assert "plan" in event_types
    assert "task" in event_types
    task_events = [e for e in events if e["type"] == "task"]
    assert any(e["status"] == "started" for e in task_events)
    assert any(e["status"] == "completed" for e in task_events)
    assert "log" in event_types
    assert "test" in event_types
    assert "done" in event_types


async def test_stop_cancellation():
    client = _FakeClient()
    orchestrator = Orchestrator(goal="test", client=client)
    events = []

    def on_event(event):
        events.append(event)
        if event["type"] == "cycle":
            orchestrator.stop()

    orchestrator.on_event = on_event
    await orchestrator.run()

    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["reason"] == "stopped"


async def test_task_failure_marked():
    client = _FakeClient(test='{"passed": true, "summary": "ok", "failures": []}')
    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, patch(
        "furrow.core.orchestrator.TesterAgent"
    ) as MockTester:
        async def bad_run():
            raise RuntimeError("fail")

        MockWorker.return_value.run = bad_run
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        orchestrator = Orchestrator(goal="test", client=client)
        events = []

        def on_event(event):
            events.append(event)

        orchestrator.on_event = on_event
        await orchestrator.run()

    assert orchestrator.tasks[0].status == "failed"
    task_events = [e for e in events if e["type"] == "task"]
    assert any(e["status"] == "failed" for e in task_events)
    assert orchestrator._is_done() is False


async def test_done_when_all_complete():
    client = _FakeClient(test='{"passed": true, "summary": "ok", "failures": []}')
    with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        orchestrator = Orchestrator(goal="test", client=client)
        events = []

        def on_event(event):
            events.append(event)

        orchestrator.on_event = on_event
        await orchestrator.run()

    assert orchestrator._is_done() is True
    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["reason"] == "complete"
