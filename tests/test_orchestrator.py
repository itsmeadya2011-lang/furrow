import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakePlanner:
    def __init__(self, plan):
        self._plan = plan

    async def plan(self, goal):
        return self._plan


class FakeTester:
    def __init__(self, result):
        self._result = result

    async def run(self, goal, tasks):
        return self._result


class FakeLLMClient:
    def __init__(self):
        self.calls = []

    async def complete(self, prompt, system=None, model=None):
        return "ok"

    async def read_file(self, path):
        return ""

    async def write_file(self, path, content):
        self.calls.append((path, content))

    def list_files(self, directory):
        return []


def _make_orchestrator(plan, test_result=None, settings=None):
    client = FakeLLMClient()
    orch = Orchestrator(goal="goal", client=client, settings=settings or Settings())
    orch.planner = FakePlanner(plan)
    orch._test_result = test_result
    return orch, client


@pytest.mark.asyncio
async def test_orchestrator_done_no_tasks():
    plan = Plan(tasks=[], rationale="done")
    orch, _ = _make_orchestrator(plan)
    with patch("furrow.core.orchestrator.TesterAgent", lambda client=None, settings=None: FakeTester(TestResult(passed=True, summary="ok"))):
        with patch("furrow.core.orchestrator.WorkerAgent", lambda task, client=None, settings=None: None):
            await orch._cycle()
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_not_done_with_pending_tasks():
    tasks = [TaskModel(id="1", description="t", status="pending")]
    plan = Plan(tasks=tasks, rationale="x")
    orch, _ = _make_orchestrator(plan)
    # don't run cycle; set tasks directly
    orch.tasks = tasks
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_done_when_all_completed_and_tests_pass():
    tasks = [TaskModel(id="1", description="t", status="completed")]
    orch = Orchestrator(goal="g", client=FakeLLMClient(), settings=Settings())
    orch.tasks = tasks
    orch.last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_not_done_when_tests_fail():
    tasks = [TaskModel(id="1", description="t", status="completed")]
    orch = Orchestrator(goal="g", client=FakeLLMClient(), settings=Settings())
    orch.tasks = tasks
    orch.last_test_result = TestResult(passed=False, summary="fail")
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_max_cycles_enforced():
    plan = Plan(tasks=[TaskModel(id="1", description="t")], rationale="x")

    class Worker:
        def __init__(self, task, client=None, settings=None):
            self.task = task

        async def run(self):
            self.task.status = "completed"
            return "done"

    # Tests keep failing so the loop never self-terminates; max_cycles must stop it.
    tester_result = TestResult(passed=False, summary="fail", failures=["boom"])

    def tester_factory(client=None, settings=None):
        return FakeTester(tester_result)

    settings = Settings(max_cycles=2)
    client = FakeLLMClient()
    orch = Orchestrator(goal="g", client=client, settings=settings)
    orch.planner = FakePlanner(plan)
    with patch("furrow.core.orchestrator.WorkerAgent", Worker):
        with patch("furrow.core.orchestrator.TesterAgent", tester_factory):
            await orch.run()
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_on_event_callback():
    plan = Plan(tasks=[], rationale="done")
    events = []
    settings = Settings()
    client = FakeLLMClient()
    orch = Orchestrator(goal="g", client=client, settings=settings, on_event=events.append)
    orch.planner = FakePlanner(plan)
    with patch("furrow.core.orchestrator.TesterAgent", lambda client=None, settings=None: FakeTester(TestResult(passed=True, summary="ok"))):
        with patch("furrow.core.orchestrator.WorkerAgent", lambda task, client=None, settings=None: None):
            await orch._cycle()
    assert len(events) >= 1