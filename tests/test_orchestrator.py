from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, settings
from furrow.core.orchestrator import Orchestrator
from tests.conftest import MockLLMClient

PLANNER_RESP = (
    '{"tasks": [{"id": "1", "description": "do it", "files": [], "dependencies": []}], "rationale": "ok"}'
)
WORKER_RESP = '{"summary": "done", "edits": [], "issues": []}'
TESTER_PASS = '{"passed": true, "summary": "all good", "failures": []}'
TESTER_FAIL = '{"passed": false, "summary": "broken", "failures": ["test_x failed"]}'
EMPTY_PLAN = '{"tasks": [], "rationale": "nothing to do"}'


def _build_orch(
    responses: list[str],
    on_progress=None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    max_cycles: int = 0,
) -> tuple[Orchestrator, MockLLMClient]:
    client = MockLLMClient(responses=responses)
    orch = Orchestrator(goal="test goal", client=client, on_progress=on_progress)
    orch.tester._run_tests = AsyncMock(return_value="")
    if monkeypatch is not None:
        monkeypatch.setattr(settings, "max_cycles", max_cycles)
    return orch, client


def _msg_str(msg) -> str:
    return str(msg)


class TestIsDone:
    def test_all_completed_and_tests_passed(self):
        orch = Orchestrator(goal="g", client=MockLLMClient())
        orch._tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
        for t in orch._tasks:
            t.status = "completed"
        orch._last_test_passed = True
        assert orch._is_done() is True

    def test_all_completed_and_tests_failed(self):
        orch = Orchestrator(goal="g", client=MockLLMClient())
        orch._tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
        for t in orch._tasks:
            t.status = "completed"
        orch._last_test_passed = False
        assert orch._is_done() is False

    def test_has_failed_task(self):
        orch = Orchestrator(goal="g", client=MockLLMClient())
        orch._tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed", result="boom"),
        ]
        orch._last_test_passed = True
        assert orch._is_done() is False

    def test_empty_tasks(self):
        orch = Orchestrator(goal="g", client=MockLLMClient())
        orch._tasks = []
        orch._last_test_passed = True
        assert orch._is_done() is True

    def test_partial_completion(self):
        orch = Orchestrator(goal="g", client=MockLLMClient())
        orch._tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ]
        orch._last_test_passed = True
        assert orch._is_done() is False

    def test_last_test_passed_defaults_false(self):
        orch = Orchestrator(goal="g", client=MockLLMClient())
        orch._tasks = [TaskModel(id="1", description="a", status="completed")]
        assert orch._last_test_passed is False
        assert orch._is_done() is False


class TestMaxCycles:
    async def test_max_cycles_stops_loop_with_failing_tests(self, monkeypatch):
        orch, _ = _build_orch(
            [PLANNER_RESP, WORKER_RESP, TESTER_FAIL] * 3, monkeypatch=monkeypatch, max_cycles=3
        )
        await orch.run()
        assert orch.cycles == 3
        assert orch._last_test_passed is False

    async def test_max_cycles_1_stops_immediately(self, monkeypatch):
        orch, _ = _build_orch([PLANNER_RESP, WORKER_RESP, TESTER_FAIL], monkeypatch=monkeypatch, max_cycles=1)
        await orch.run()
        assert orch.cycles == 1

    async def test_stops_when_tests_pass_before_max_cycles(self, monkeypatch):
        orch, _ = _build_orch([PLANNER_RESP, WORKER_RESP, TESTER_PASS], monkeypatch=monkeypatch, max_cycles=5)
        await orch.run()
        assert orch.cycles == 1
        assert orch._last_test_passed is True

    async def test_no_max_cycles_terminates_via_tests_pass(self):
        # max_cycles=0 (disabled): passing tests must still halt the loop.
        orch, _ = _build_orch([PLANNER_RESP, WORKER_RESP, TESTER_PASS])
        await orch.run()
        assert orch.cycles == 1


class TestOnProgress:
    async def test_sync_callback_called(self, monkeypatch):
        messages: list = []
        orch, _ = _build_orch(
            [PLANNER_RESP, WORKER_RESP, TESTER_PASS],
            on_progress=messages.append,
            monkeypatch=monkeypatch,
            max_cycles=5,
        )
        await orch.run()
        assert messages, "on_progress callback should have been called"
        joined = " ".join(_msg_str(m) for m in messages)
        assert "Cycle 1" in joined
        assert "Goal complete" in joined

    async def test_async_callback_called(self, monkeypatch):
        messages: list = []

        async def collector(msg) -> None:
            messages.append(msg)

        orch, _ = _build_orch(
            [PLANNER_RESP, WORKER_RESP, TESTER_PASS],
            on_progress=collector,
            monkeypatch=monkeypatch,
            max_cycles=5,
        )
        await orch.run()
        await asyncio.sleep(0)
        assert messages, "async on_progress callback should have been called"
        joined = " ".join(_msg_str(m) for m in messages)
        assert "Goal complete" in joined


class TestFullCycle:
    async def test_cycle_passes_and_stops(self, monkeypatch):
        orch, client = _build_orch([PLANNER_RESP, WORKER_RESP, TESTER_PASS], monkeypatch=monkeypatch, max_cycles=5)
        await orch.run()
        assert orch.cycles == 1
        assert len(orch._tasks) == 1
        assert orch._tasks[0].status == "completed"
        assert orch._last_test_passed is True
        assert client.call_count == 3  # planner + worker + tester

    async def test_failed_tests_do_not_short_circuit(self, monkeypatch):
        orch, client = _build_orch(
            [PLANNER_RESP, WORKER_RESP, TESTER_FAIL] * 2, monkeypatch=monkeypatch, max_cycles=2
        )
        await orch.run()
        assert orch.cycles == 2
        assert client.call_count == 6
        assert orch._is_done() is False
        assert orch._last_test_passed is False

    async def test_goal_updated_when_tests_fail(self, monkeypatch):
        orch, _ = _build_orch([PLANNER_RESP, WORKER_RESP, TESTER_FAIL], monkeypatch=monkeypatch, max_cycles=1)
        original_goal = orch.goal
        await orch.run()
        assert orch.goal != original_goal
        assert "Fix failing tests" in orch.goal

    async def test_no_tasks_planned(self, monkeypatch):
        orch, _ = _build_orch([EMPTY_PLAN], monkeypatch=monkeypatch, max_cycles=5)
        await orch._cycle()
        assert orch._tasks == []
        assert orch._last_test_passed is False

    async def test_worker_exception_marks_task_failed(self, monkeypatch):
        async def boom(self) -> str:
            raise RuntimeError("worker exploded")

        monkeypatch.setattr(WorkerAgent, "run", boom)
        orch, _ = _build_orch([PLANNER_RESP, TESTER_PASS], monkeypatch=monkeypatch, max_cycles=5)
        await orch._cycle()
        assert any(t.status == "failed" for t in orch._tasks)
        assert orch._tasks[0].result is not None
        assert "worker exploded" in orch._tasks[0].result
        assert orch._is_done() is False
