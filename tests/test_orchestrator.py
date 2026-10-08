import json

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


class FakeLLMClient:
    """Queue of canned responses (or exceptions) returned by complete()."""

    def __init__(self, responses: list) -> None:
        self.settings = Settings()
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class EventCollector:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def __call__(self, message: str) -> None:
        self.messages.append(message)


@pytest.fixture(autouse=True)
def fake_test_runner(monkeypatch):
    """Keep orchestrator tests from launching the real test suite."""

    async def _fake_run_tests(self) -> str:
        return "Fake test output (no tests run)"

    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)


@pytest.mark.asyncio
async def test_cycle_completes_and_halts_when_no_more_tasks():
    fake = FakeLLMClient(
        [
            json.dumps(
                {"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}
            ),
            "worker output",
            json.dumps({"passed": True, "summary": "ok", "failures": []}),
            json.dumps({"tasks": [], "rationale": "nothing left"}),
        ]
    )
    events = EventCollector()
    orchestrator = Orchestrator(goal="build a thing", client=fake, on_event=events)
    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator._get_tasks()[0].status == "completed"
    assert orchestrator._get_tasks()[0].result == "worker output"
    assert any("Task 1 completed" in m for m in events.messages)
    assert any("Tests passed" in m for m in events.messages)
    assert any("Goal complete" in m for m in events.messages)


@pytest.mark.asyncio
async def test_task_tracking_fixes_is_done_bug():
    # Regression: _get_tasks() used to always return [], which made the
    # orchestrator halt after a single cycle regardless of task state.
    fake = FakeLLMClient(
        [
            json.dumps(
                {"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}
            ),
            "worker output",
            json.dumps({"passed": True, "summary": "ok", "failures": []}),
        ]
    )
    orchestrator = Orchestrator(goal="g", client=fake)
    await orchestrator._cycle()
    tasks = orchestrator._get_tasks()
    assert tasks != []
    assert tasks[0].status == "completed"
    # A completed cycle is not the end: the planner must confirm the goal
    # is complete by returning no tasks in the next cycle.
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_halt_when_planner_declares_complete():
    fake = FakeLLMClient(
        [
            json.dumps({"tasks": [], "rationale": "goal already achieved"}),
        ]
    )
    events = EventCollector()
    orchestrator = Orchestrator(goal="g", client=fake, on_event=events)
    await orchestrator.run()
    assert orchestrator.cycles == 1
    assert any("Goal complete" in m for m in events.messages)


@pytest.mark.asyncio
async def test_failed_task_keeps_loop_running():
    fake = FakeLLMClient(
        [
            json.dumps(
                {"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}
            ),
            RuntimeError("worker exploded"),
            json.dumps({"passed": False, "summary": "bad", "failures": ["assertion"]}),
        ]
    )
    orchestrator = Orchestrator(goal="g", client=fake)
    await orchestrator._cycle()
    task = orchestrator._get_tasks()[0]
    assert task.status == "failed"
    assert "worker exploded" in task.result
    assert orchestrator._is_done() is False
    # Original goal is preserved when tests fail.
    assert "g" in orchestrator.goal
    assert "Fix the following failing tests" in orchestrator.goal
    assert "assertion" in orchestrator.goal


@pytest.mark.asyncio
async def test_max_cycles_honored():
    fake = FakeLLMClient(
        [
            json.dumps(
                {"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}
            ),
            RuntimeError("worker exploded"),
            json.dumps({"passed": False, "summary": "bad", "failures": ["boom"]}),
        ]
    )
    events = EventCollector()
    orchestrator = Orchestrator(
        goal="g", client=fake, settings=Settings(max_cycles=1), on_event=events
    )
    await orchestrator.run()
    assert orchestrator.cycles == 1
    assert any("max_cycles" in m for m in events.messages)


@pytest.mark.asyncio
async def test_record_tasks_updates_by_id():
    orchestrator = Orchestrator(goal="g")
    first = TaskModel(id="1", description="first")
    second = TaskModel(id="1", description="first v2")
    third = TaskModel(id="2", description="other")
    orchestrator._record_tasks([first])
    orchestrator._record_tasks([second, third])
    assert len(orchestrator._get_tasks()) == 2
    by_id = {t.id: t for t in orchestrator._get_tasks()}
    assert by_id["1"].description == "first v2"
    assert by_id["2"].description == "other"


@pytest.mark.asyncio
async def test_parallelism_is_limited_by_semaphore():
    orchestrator = Orchestrator(goal="g", settings=Settings(max_parallel_tasks=1))
    async with orchestrator._semaphore:
        assert orchestrator._semaphore.locked()
    assert not orchestrator._semaphore.locked()