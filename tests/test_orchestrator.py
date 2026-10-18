from furrow.config import Plan
from furrow.core.orchestrator import Orchestrator


def test_get_tasks_empty():
    orch = Orchestrator(goal="anything")
    assert orch._get_tasks() == []


def test_is_done_with_no_tasks():
    orch = Orchestrator(goal="anything")
    assert orch._is_done() is True


async def test_cycle_with_empty_plan(monkeypatch):
    orch = Orchestrator(goal="do nothing")

    async def fake_plan(goal):
        return Plan(tasks=[], rationale="already complete")

    orch.planner.plan = fake_plan
    await orch._cycle()
    assert orch.cycles == 0
    assert orch._is_done() is True
