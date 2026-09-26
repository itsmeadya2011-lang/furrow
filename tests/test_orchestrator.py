import json

from furrow.agents.planner import PlannerAgent
from furrow.config import TestResult
from furrow.core.orchestrator import Orchestrator
from tests.fakes import FakeClient, FakeTester


def make_plan_json(num_tasks: int) -> str:
    payload = {
        "tasks": [
            {"id": str(i + 1), "description": f"task {i + 1}", "files": [], "dependencies": []}
            for i in range(num_tasks)
        ],
        "rationale": "ok",
    }
    return json.dumps(payload)


async def test_orchestrator_stops_when_tests_pass():
    fake = FakeClient(responses=[make_plan_json(1), "done task 1"])
    tester = FakeTester(TestResult(passed=True, summary="all good", failures=[]))
    orch = Orchestrator(
        goal="build x", client=fake, planner=PlannerAgent(client=fake), tester=tester
    )
    await orch.run()
    assert orch.cycles == 1
    assert tester.calls == 1
    assert orch.last_test_result.passed is True
    assert orch.tasks[0].status == "completed"


async def test_orchestrator_respects_max_cycles():
    fake = FakeClient(responses=[make_plan_json(1), "done task 1"])
    tester = FakeTester(TestResult(passed=False, summary="fail", failures=["broken"]))
    orch = Orchestrator(
        goal="build x", client=fake, planner=PlannerAgent(client=fake), tester=tester
    )
    orch.settings.max_cycles = 1
    await orch.run()
    assert orch.cycles == 1
    assert tester.calls == 1


async def test_orchestrator_parallel_tasks(tmp_path):
    fake = FakeClient(responses=[make_plan_json(2), "done a", "done b"])
    tester = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    orch = Orchestrator(
        goal="build x", client=fake, planner=PlannerAgent(client=fake), tester=tester
    )
    orch.settings.max_parallel_tasks = 1
    orch.settings.workspace = tmp_path
    await orch.run()
    assert orch.cycles == 1
    assert all(t.status == "completed" for t in orch.tasks)


async def test_orchestrator_handles_empty_plan():
    fake = FakeClient(responses=[json.dumps({"tasks": [], "rationale": "done"})])
    tester = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    orch = Orchestrator(
        goal="build x", client=fake, planner=PlannerAgent(client=fake), tester=tester
    )
    await orch.run()
    assert orch.cycles == 1
    assert tester.calls == 0
    assert orch.last_test_result is None
