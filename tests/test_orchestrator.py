from __future__ import annotations

import asyncio

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.agents.tester import TesterAgent


@pytest.fixture(autouse=True)
def _stub_test_runner(monkeypatch):
    """Prevent TesterAgent from spawning a real pytest subprocess during tests."""
    async def _fake_run_tests(self) -> str:
        return "No test runner available in test environment."
    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)


class MockLLMClient:
    """A mock LLM client that returns scripted responses."""

    def __init__(self):
        self.settings = Settings()
        self.responses = []
        self.write_calls = []

    def queue(self, response: str):
        self.responses.append(response)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if self.responses:
            return self.responses.pop(0)
        return '{"tasks": [], "rationale": "done"}'

    async def write_file(self, path: str, content: str) -> None:
        self.write_calls.append((path, content))


PLAN_PASS = '{"tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}], "rationale": "test"}'
WORKER_SUMMARY = "SUMMARY: Did the thing."
TEST_PASS = '{"passed": true, "summary": "all good", "failures": []}'
TEST_FAIL = '{"passed": false, "summary": "fail", "failures": ["bug in line 42"]}'


def test_orchestrator_done_after_cycle():
    """Orchestrator halts after one cycle when tasks complete and tests pass."""
    client = MockLLMClient()
    client.queue(PLAN_PASS)
    client.queue(WORKER_SUMMARY)
    client.queue(TEST_PASS)

    orch = Orchestrator(goal="test goal", client=client)
    asyncio.run(orch.run())
    assert orch.cycles == 1
    assert orch._is_done() is True


def test_orchestrator_not_done_with_failed_task():
    """Orchestrator is not done if a task failed."""
    client = MockLLMClient()
    client.queue(PLAN_PASS)
    client.queue(WORKER_SUMMARY)
    client.queue(TEST_PASS)

    orch = Orchestrator(goal="test", client=client)
    asyncio.run(orch._cycle())
    # Force a task to failed
    orch._current_tasks[0].status = "failed"
    assert orch._is_done() is False


def test_orchestrator_preserves_original_goal():
    """Original goal is preserved even when working goal changes."""
    client = MockLLMClient()
    client.queue(PLAN_PASS)
    client.queue(WORKER_SUMMARY)
    client.queue(TEST_FAIL)

    orch = Orchestrator(goal="build feature X", client=client)
    asyncio.run(orch._cycle())
    assert orch.original_goal == "build feature X"
    assert orch.goal.startswith("Fix failing tests")


def test_orchestrator_max_cycles():
    """max_cycles limits the loop."""
    client = MockLLMClient()
    # Queue 2 cycles worth of responses (planner + worker + tester each)
    client.queue(PLAN_PASS)
    client.queue(WORKER_SUMMARY)
    client.queue(TEST_FAIL)
    client.queue(PLAN_PASS)
    client.queue(WORKER_SUMMARY)
    client.queue(TEST_FAIL)

    orch = Orchestrator(goal="test", client=client)
    orch.client.settings.max_cycles = 2
    asyncio.run(orch.run())
    assert orch.cycles == 2


def test_orchestrator_emits_events():
    """on_event callback is called with event dicts."""
    events = []
    client = MockLLMClient()
    client.queue(PLAN_PASS)
    client.queue(WORKER_SUMMARY)
    client.queue(TEST_PASS)

    orch = Orchestrator(goal="test", client=client, on_event=events.append)
    asyncio.run(orch.run())
    assert any(e.get("type") == "start" for e in events)
    assert any(e.get("type") == "plan" for e in events)
    assert any(e.get("type") == "complete" for e in events)