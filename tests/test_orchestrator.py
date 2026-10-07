from __future__ import annotations

import asyncio

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class _DummySettings(Settings):
    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)


class FakeClient(LLMClient):
    """LLMClient with overridable async methods for deterministic tests."""

    def __init__(self, settings=None, plan=None, test_result=None, worker_result="done", **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(settings=settings or _DummySettings(provider=Provider.ANTHROPIC, anthropic_api_key="x"))
        self._plan = plan
        self._test_result = test_result
        self._worker_result = worker_result
        self.planner_calls = 0
        self.tester_calls = 0

    async def complete(self, prompt, system="", model=None):  # type: ignore[override]
        return ""

    async def complete_json(self, prompt, system="", model=None):  # type: ignore[override]
        import json
        return json.dumps({"edits": [], "summary": self._worker_result})


def _plan_with(tasks: list[TaskModel]) -> Plan:
    return Plan(tasks=tasks, rationale="test")


def test_get_tasks_returns_stored_tasks():
    client = FakeClient()
    orch = Orchestrator(goal="g", client=client)
    assert orch._get_tasks() == []
    tasks = [TaskModel(id="1", description="do thing")]
    orch.tasks = tasks
    assert orch._get_tasks() is tasks


def test_is_done_when_all_completed():
    client = FakeClient()
    orch = Orchestrator(goal="g", client=client)
    orch.tasks = [TaskModel(id="1", description="a", status="completed")]
    assert orch._is_done() is True


def test_is_done_false_when_any_failed():
    client = FakeClient()
    orch = Orchestrator(goal="g", client=client)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_false_when_pending():
    client = FakeClient()
    orch = Orchestrator(goal="g", client=client)
    orch.tasks = [TaskModel(id="1", description="a", status="pending")]
    assert orch._is_done() is False


def test_max_cycles_halts(monkeypatch):
    # Avoid real planner/worker/tester by patching the classes used in orchestrator.
    # Worker keeps failing so _is_done() never triggers and the failure guard
    # (needs 4 consecutive failure cycles) is not reached before max_cycles.
    import furrow.core.orchestrator as orch_mod

    class FakePlanner:
        def __init__(self, client):
            self.client = client

        async def plan(self, goal):
            return _plan_with([TaskModel(id="1", description="x")])

    class FakeWorker:
        def __init__(self, task, client):
            self.task = task
            self.client = client

        async def run(self):
            raise RuntimeError("boom")

    class FakeTester:
        def __init__(self, client):
            self.client = client

        async def run(self, goal, tasks):
            return TestResult(passed=False, summary="fail", failures=["fail"])

    monkeypatch.setattr(orch_mod, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(orch_mod, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", FakeTester)

    settings = _DummySettings(provider=Provider.ANTHROPIC, anthropic_api_key="x", max_cycles=2)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="g", client=client)
    asyncio.run(orch.run())
    # max_cycles=2 -> runs cycles 1 and 2, halts at top of cycle 3.
    assert orch.cycles == 3


def test_repeated_failures_halts(monkeypatch):
    import furrow.core.orchestrator as orch_mod

    class FakePlanner:
        def __init__(self, client):
            self.client = client

        async def plan(self, goal):
            return _plan_with([TaskModel(id="1", description="x", status="pending")])

    class FakeWorker:
        def __init__(self, task, client):
            self.task = task
            self.client = client

        async def run(self):
            raise RuntimeError("boom")

    class FakeTester:
        def __init__(self, client):
            self.client = client

        async def run(self, goal, tasks):
            return TestResult(passed=False, summary="fail", failures=["fail"])

    monkeypatch.setattr(orch_mod, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(orch_mod, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", FakeTester)

    settings = _DummySettings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="g", client=client)
    asyncio.run(orch.run())
    # 3 failure-only cycles then halt
    assert orch.failed_cycles > 3