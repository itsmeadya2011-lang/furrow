from __future__ import annotations

from furrow.config import TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from tests._fakes import CollectingReporter, FakeClient, make_settings

PLAN = (
    '{"tasks":[{"id":"1","description":"t1","files":["a.py"],"dependencies":[]},'
    '{"id":"2","description":"t2","files":[],"dependencies":[]}],"rationale":"r"}'
)
PLAN4 = (
    '{"tasks":[{"id":"1","description":"t1","files":["a.py"],"dependencies":[]},'
    '{"id":"2","description":"t2","files":[]},'
    '{"id":"3","description":"t3","files":[]},'
    '{"id":"4","description":"t4","files":[]}],"rationale":"r"}'
)
WORKER = '{"summary":"done","files":[{"path":"a.py","content":"print(1)"}]}'
TEST_OK = '{"passed":true,"summary":"all good","failures":[]}'
TEST_FAIL = '{"passed":false,"summary":"broken","failures":["x failed"]}'


def _make_complete_fn(plan=PLAN, worker=WORKER, tester=TEST_OK, worker_exc=None):
    async def complete(prompt, system="", model=None):
        if "planning agent" in prompt:
            return plan
        if "tester agent" in prompt:
            return tester
        if worker_exc is not None:
            raise worker_exc
        return worker

    return complete


async def test_orchestrator_success_single_cycle(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(settings, responses=[PLAN, WORKER, WORKER, TEST_OK])
    orch = Orchestrator(goal="g", client=client, reporter=CollectingReporter())
    await orch.run()
    assert orch.cycles == 1
    assert orch._tasks[0].status == "completed"
    assert orch._tasks[1].status == "completed"
    assert (settings.workspace / "a.py").read_text() == "print(1)"
    assert orch._last_test_result.passed is True


async def test_orchestrator_test_failure_retries(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(
        settings,
        complete_fn=_make_complete_fn(tester=TEST_FAIL),
    )
    orch = Orchestrator(goal="g", client=client, reporter=CollectingReporter(), max_cycles=2)
    await orch.run()
    # Two cycles: each fails tests, goal rewritten; capped at 2.
    assert orch.cycles == 2


async def test_orchestrator_max_cycles_unlimited_stops_when_done(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(settings, responses=[PLAN, WORKER, WORKER, TEST_OK])
    orch = Orchestrator(goal="g", client=client, reporter=CollectingReporter())
    await orch.run()
    assert orch.cycles == 1


async def test_orchestrator_max_cycles_zero_means_unlimited(tmp_path):
    # max_cycles=0 + tests always failing would loop forever; give it a 1-task plan
    # and verify the cap does NOT trigger when max_cycles=0 (it is unlimited).
    settings = make_settings(tmp_path)
    client = FakeClient(settings, responses=[PLAN, WORKER, WORKER, TEST_OK])
    orch = Orchestrator(goal="g", client=client, reporter=CollectingReporter(), max_cycles=0)
    await orch.run()
    assert orch.cycles == 1


async def test_orchestrator_failed_task_does_not_finish(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(
        settings,
        complete_fn=_make_complete_fn(worker_exc=RuntimeError("boom")),
    )
    orch = Orchestrator(goal="g", client=client, reporter=CollectingReporter(), max_cycles=1)
    await orch.run()
    assert orch.cycles == 1
    assert orch._tasks[0].status == "failed"
    assert orch._is_done() is False


async def test_orchestrator_enforces_max_parallel(tmp_path):
    settings = make_settings(tmp_path, max_parallel_tasks=2)
    client = FakeClient(
        settings, delay=0.05, complete_fn=_make_complete_fn(plan=PLAN4, tester=TEST_OK)
    )
    orch = Orchestrator(
        goal="g", client=client, reporter=CollectingReporter(), max_parallel_tasks=2
    )
    await orch.run()
    assert client.max_concurrent <= 2


def test_is_done_empty_tasks(tmp_path):
    orch = Orchestrator(
        goal="g", client=FakeClient(make_settings(tmp_path)), reporter=CollectingReporter()
    )
    orch._tasks = []
    assert orch._is_done() is True


def test_is_done_all_completed_tests_pass(tmp_path):
    orch = Orchestrator(
        goal="g", client=FakeClient(make_settings(tmp_path)), reporter=CollectingReporter()
    )
    t = TaskModel(id="1", description="d")
    t.status = "completed"
    orch._tasks = [t]
    orch._last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is True


def test_is_done_all_completed_tests_fail(tmp_path):
    orch = Orchestrator(
        goal="g", client=FakeClient(make_settings(tmp_path)), reporter=CollectingReporter()
    )
    t = TaskModel(id="1", description="d")
    t.status = "completed"
    orch._tasks = [t]
    orch._last_test_result = TestResult(passed=False, summary="bad", failures=["x"])
    assert orch._is_done() is False


def test_is_done_failed_task(tmp_path):
    orch = Orchestrator(
        goal="g", client=FakeClient(make_settings(tmp_path)), reporter=CollectingReporter()
    )
    t = TaskModel(id="1", description="d")
    t.status = "failed"
    orch._tasks = [t]
    assert orch._is_done() is False


def test_is_done_pending_task(tmp_path):
    orch = Orchestrator(
        goal="g", client=FakeClient(make_settings(tmp_path)), reporter=CollectingReporter()
    )
    t = TaskModel(id="1", description="d")
    t.status = "pending"
    orch._tasks = [t]
    assert orch._is_done() is False


def test_get_tasks_returns_stored(tmp_path):
    orch = Orchestrator(
        goal="g", client=FakeClient(make_settings(tmp_path)), reporter=CollectingReporter()
    )
    t = TaskModel(id="1", description="d")
    t.status = "completed"
    orch._tasks = [t]
    assert orch._get_tasks() == [t]
