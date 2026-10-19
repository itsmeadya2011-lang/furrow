import asyncio
from unittest.mock import AsyncMock, MagicMock

from furrow.config import Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def make_orch(tasks=None):
    s = Settings(_env_file=None, max_cycles=0, max_parallel_tasks=2)
    o = Orchestrator(goal="g", settings=s)
    if tasks is not None:
        o.tasks = tasks
    return o


def test_instantiate():
    o = make_orch()
    assert o.goal == "g"
    assert o.tasks == []


def test_is_done_no_tasks():
    o = make_orch([])
    assert o._is_done() is False


def test_is_done_all_completed():
    tasks = [TaskModel(id="1", description="a", status="completed")]
    o = make_orch(tasks)
    assert o._is_done() is True


def test_is_done_some_failed():
    tasks = [TaskModel(id="1", description="a", status="failed")]
    o = make_orch(tasks)
    assert o._is_done() is False


def test_max_cycles_enforced():
    s = Settings(_env_file=None, max_cycles=1, max_parallel_tasks=2)
    o = Orchestrator(goal="g", settings=s)
    o.planner = MagicMock()
    o.planner.plan = AsyncMock(
        return_value=__import__("furrow.config", fromlist=["Plan"]).Plan(tasks=[], rationale="")
    )
    o.tester = MagicMock()
    o.tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))
    asyncio.get_event_loop().run_until_complete(o.run())
    assert o.cycles == 1


def test_tasks_persist_across_cycles():
    from furrow.config import Plan

    s = Settings(_env_file=None, max_cycles=2, max_parallel_tasks=2)
    o = Orchestrator(goal="g", settings=s)
    plan1 = Plan(tasks=[TaskModel(id="1", description="a")], rationale="")
    plan2 = Plan(tasks=[TaskModel(id="2", description="b")], rationale="")
    o.planner = MagicMock()
    o.planner.plan = AsyncMock(side_effect=[plan1, plan2])
    o.tester = MagicMock()
    o.tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))
    # worker agents created inside; mock WorkerAgent? patch
    import furrow.core.orchestrator as oc

    orig = oc.WorkerAgent

    class FakeWorker:
        def __init__(self, task, client=None, settings=None):
            self.task = task

        async def run(self):
            return "done"

    oc.WorkerAgent = FakeWorker
    try:
        asyncio.get_event_loop().run_until_complete(o.run())
    finally:
        oc.WorkerAgent = orig
    ids = {t.id for t in o.tasks}
    assert "1" in ids and "2" in ids


def test_failed_tasks_trigger_retry():
    from furrow.config import Plan

    s = Settings(_env_file=None, max_cycles=2, max_parallel_tasks=2)
    o = Orchestrator(goal="g", settings=s)
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="")
    o.planner = MagicMock()
    o.planner.plan = AsyncMock(return_value=plan)
    o.tester = MagicMock()
    o.tester.run = AsyncMock(
        return_value=TestResult(passed=False, summary="fail", failures=["boom"])
    )
    import furrow.core.orchestrator as oc

    class FakeWorker:
        def __init__(self, task, client=None, settings=None):
            self.task = task

        async def run(self):
            raise RuntimeError("boom")

    oc.WorkerAgent = FakeWorker
    try:
        asyncio.get_event_loop().run_until_complete(o.run())
    finally:
        oc.WorkerAgent = (
            FakeWorker.__class__
            and __import__("furrow.agents.worker", fromlist=["WorkerAgent"]).WorkerAgent
        )
    assert o.cycles == 2
