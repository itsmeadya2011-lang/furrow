"""Orchestrator loop regression tests.

Every LLM client here is a fake and both agent types are patched out, so these
tests never touch the network and never spawn a worker/tester subprocess.
"""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from furrow.config import Settings, TaskModel, TestResult
from furrow.core import orchestrator as orch_mod
from furrow.core.orchestrator import Orchestrator

TIMEOUT = 10


def plan_payload(count):
    """Build a valid planner response containing ``count`` independent tasks."""
    return json.dumps(
        {
            "tasks": [
                {"id": str(i), "description": f"implement slice {i}", "files": [f"mod{i}.py"]}
                for i in range(1, count + 1)
            ],
            "rationale": "one parallel slice",
        }
    )


def dependency_plan(spec):
    """Build a valid planner response whose tasks carry explicit dependency ids.

    ``spec`` maps each task id to its dependency ids; iteration order becomes the
    order the planner emits the tasks in.
    """
    return json.dumps(
        {
            "tasks": [
                {
                    "id": task_id,
                    "description": f"implement slice {task_id}",
                    "files": [f"mod{task_id}.py"],
                    "dependencies": list(deps),
                }
                for task_id, deps in spec.items()
            ],
            "rationale": "dependency aware slices",
        }
    )


class FakeLLMClient:
    """Stand-in for ``LLMClient``: records calls and replays canned text."""

    def __init__(self, responses=(), fallback=None, **settings_overrides):
        self.responses = list(responses)
        self.fallback = fallback if fallback is not None else plan_payload(1)
        self.calls = []
        self.settings = Settings(_env_file=None, **settings_overrides)

    async def complete(self, prompt, system="", model=None):
        self.calls.append((prompt, model))
        if self.responses:
            return self.responses.pop(0)
        return self.fallback


def _worker_stub(stubs):
    class StubWorkerAgent:
        def __init__(self, task, client=None, settings=None):
            self.task = task
            self.client = client

        async def run(self):
            stubs.active += 1
            stubs.peak = max(stubs.peak, stubs.active)
            try:
                await asyncio.sleep(stubs.worker_delay)
                if self.task.id in stubs.worker_fail_ids:
                    raise RuntimeError(f"worker failed on task {self.task.id}")
                return f"implemented task {self.task.id}"
            finally:
                stubs.active -= 1

    return StubWorkerAgent


def _tester_stub(stubs):
    class StubTesterAgent:
        def __init__(self, client=None, settings=None):
            self.client = client

        async def run(self, goal, tasks):
            stubs.tester_calls.append((goal, [task.id for task in tasks]))
            if len(stubs.tester_results) > 1:
                return stubs.tester_results.pop(0)
            return stubs.tester_results[0]

    return StubTesterAgent


@pytest.fixture
def agents(monkeypatch):
    """Patch both agent types so no worker/tester call ever reaches an LLM."""
    stubs = SimpleNamespace(
        worker_delay=0.0,
        worker_fail_ids=(),
        tester_results=[TestResult(passed=True, summary="all good", failures=[])],
        tester_calls=[],
        active=0,
        peak=0,
    )
    monkeypatch.setattr(orch_mod, "WorkerAgent", _worker_stub(stubs))
    monkeypatch.setattr(orch_mod, "TesterAgent", _tester_stub(stubs))
    return stubs


def _timeline_worker_stub(stubs):
    """Worker stub that records a ``("start"|"finish", task_id)`` event timeline."""

    class TimelineWorkerAgent:
        def __init__(self, task, client=None, settings=None):
            self.task = task
            self.client = client

        async def run(self):
            task_id = self.task.id
            stubs.events.append(("start", task_id))
            stubs.active += 1
            stubs.peak = max(stubs.peak, stubs.active)
            try:
                await asyncio.sleep(stubs.worker_delay)
                stubs.events.append(("finish", task_id))
                return f"implemented task {task_id}"
            finally:
                stubs.active -= 1

    return TimelineWorkerAgent


@pytest.fixture
def timeline_agents(monkeypatch):
    """Same as ``agents``, but the worker stub also records a start/finish timeline."""
    stubs = SimpleNamespace(
        worker_delay=0.02,
        worker_fail_ids=(),
        events=[],
        tester_results=[TestResult(passed=True, summary="all good", failures=[])],
        tester_calls=[],
        active=0,
        peak=0,
    )
    monkeypatch.setattr(orch_mod, "WorkerAgent", _timeline_worker_stub(stubs))
    monkeypatch.setattr(orch_mod, "TesterAgent", _tester_stub(stubs))
    return stubs


def _task(task_id, deps=()):
    """Build a ``TaskModel`` whose only interesting field is its dependency list."""
    return TaskModel(
        id=task_id,
        description=f"implement slice {task_id}",
        dependencies=list(deps),
    )


def _layer_ids(layers):
    """Flatten dependency layers to ids so assertions stay order-explicit."""
    return [[task.id for task in layer] for layer in layers]


async def _layers_with_deadline(orchestrator, tasks):
    """Call the synchronous ``_dependency_layers`` under a hard deadline.

    ``asyncio.wait_for`` cannot interrupt a blocking call, so a layering loop that
    never terminates would hang the whole suite rather than fail one test. Running
    the call on a daemon thread keeps such a regression contained to a single
    ``TimeoutError`` failure here. If the call raises, the thread's own traceback
    is still printed and the collect step times out, so neither outcome hangs.
    """
    box = {}

    def _runner():
        box["layers"] = orchestrator._dependency_layers(tasks)

    async def _collect():
        while not box:
            await asyncio.sleep(0.002)
        return box["layers"]

    threading.Thread(target=_runner, daemon=True).start()
    return await asyncio.wait_for(_collect(), timeout=TIMEOUT)


async def test_get_tasks_returns_accumulated_tasks(agents):
    client = FakeLLMClient([plan_payload(2)])
    orchestrator = Orchestrator(goal="build it", client=client)

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    tasks = orchestrator._get_tasks()
    assert len(tasks) == 2
    assert orchestrator.all_tasks
    assert [task.id for task in tasks] == ["1", "2"]
    assert all(isinstance(task, TaskModel) for task in tasks)


async def test_is_done_false_when_any_task_failed(agents):
    agents.worker_fail_ids = ("2",)
    orchestrator = Orchestrator(goal="build it", client=FakeLLMClient([plan_payload(3)]))

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    assert [task.status for task in orchestrator.all_tasks] == [
        "completed",
        "failed",
        "completed",
    ]
    assert orchestrator._is_done() is False


async def test_is_done_true_only_when_all_completed(agents):
    orchestrator = Orchestrator(goal="build it", client=FakeLLMClient([plan_payload(2)]))

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    assert all(task.status == "completed" for task in orchestrator.all_tasks)
    assert orchestrator._is_done() is True


async def test_is_done_false_when_no_tasks_accumulated(agents):
    # Regression: an empty task list used to look "done", halting the loop after cycle 1.
    orchestrator = Orchestrator(goal="build it", client=FakeLLMClient())

    assert orchestrator.all_tasks == []
    assert orchestrator._is_done() is False


async def test_max_cycles_stops_run(agents):
    # A failed worker keeps `_is_done()` False, so only `max_cycles` can stop the loop.
    agents.worker_fail_ids = ("1",)
    agents.tester_results = [TestResult(passed=False, summary="red", failures=["boom"])]
    orchestrator = Orchestrator(goal="build it", client=FakeLLMClient(max_cycles=1))

    await asyncio.wait_for(orchestrator.run(), timeout=TIMEOUT)

    assert orchestrator.cycles == 1
    assert orchestrator._max_cycles_reached() is True


async def test_max_cycles_zero_is_unlimited(agents):
    agents.worker_fail_ids = ("1",)
    agents.tester_results = [TestResult(passed=False, summary="red", failures=["boom"])]
    orchestrator = Orchestrator(goal="build it", client=FakeLLMClient(max_cycles=0))

    assert orchestrator._max_cycles_reached() is False
    for expected in (1, 2, 3):
        await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)
        orchestrator.cycles += 1
        assert orchestrator._max_cycles_reached() is False
    assert orchestrator.cycles == 3


async def test_parallelism_capped(agents):
    agents.worker_delay = 0.01
    client = FakeLLMClient([plan_payload(5)], max_parallel_tasks=2)
    orchestrator = Orchestrator(goal="build it", client=client)

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    assert len(orchestrator.all_tasks) == 5
    assert all(task.status == "completed" for task in orchestrator.all_tasks)
    assert agents.peak >= 1
    assert agents.peak <= 2


async def test_goal_is_preserved_after_failed_tests(agents):
    agents.tester_results = [TestResult(passed=False, summary="red", failures=["boom"])]
    orchestrator = Orchestrator(goal="the original goal", client=FakeLLMClient())

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    assert orchestrator.goal == "the original goal"


async def test_focus_carries_failures_then_resets(agents):
    agents.tester_results = [
        TestResult(passed=False, summary="red", failures=["boom"]),
        TestResult(passed=True, summary="green", failures=[]),
    ]
    orchestrator = Orchestrator(goal="the original goal", client=FakeLLMClient())
    assert orchestrator.focus == "the original goal"

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)
    assert "boom" in orchestrator.focus
    assert orchestrator.goal == "the original goal"

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)
    assert orchestrator.focus == "the original goal"
    assert orchestrator.goal == "the original goal"


async def test_planner_failure_raises_after_retries(agents):
    client = FakeLLMClient(fallback="I am afraid I cannot emit JSON today.")
    orchestrator = Orchestrator(goal="build it", client=client)

    async def _instant_sleep(*_args, **_kwargs):
        return None

    cycle = asyncio.ensure_future(orchestrator._cycle())
    with mock.patch.object(asyncio, "sleep", _instant_sleep), pytest.raises(RuntimeError):
        await asyncio.wait_for(cycle, timeout=TIMEOUT)
    assert client.calls


async def test_cycle_records_task_results(agents):
    orchestrator = Orchestrator(goal="build it", client=FakeLLMClient([plan_payload(3)]))

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    tasks = orchestrator._get_tasks()
    assert len(tasks) == 3
    for task in tasks:
        assert task.status == "completed"
        assert isinstance(task.result, str)
        assert task.result


async def test_tester_receives_the_original_goal(agents):
    client = FakeLLMClient()
    orchestrator = Orchestrator(goal="the original goal", client=client)

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    assert agents.tester_calls
    goal, task_ids = agents.tester_calls[0]
    assert goal == "the original goal"
    assert task_ids == ["1"]


async def test_orchestrator_defaults_focus_to_goal(agents):
    orchestrator = Orchestrator(goal="build it", client=FakeLLMClient())

    assert orchestrator.goal == "build it"
    assert orchestrator.focus == "build it"
    assert orchestrator.cycles == 0
    assert orchestrator.all_tasks == []


async def test_dependency_layers_independent_tasks_share_one_layer():
    orchestrator = Orchestrator(goal="g", client=FakeLLMClient([]))
    tasks = [_task("1"), _task("2"), _task("3")]

    layers = await _layers_with_deadline(orchestrator, tasks)

    assert _layer_ids(layers) == [["1", "2", "3"]]


async def test_dependency_layers_chain_produces_sequential_layers():
    orchestrator = Orchestrator(goal="g", client=FakeLLMClient([]))
    tasks = [_task("A"), _task("B", ["A"]), _task("C", ["B"])]

    layers = await _layers_with_deadline(orchestrator, tasks)

    assert _layer_ids(layers) == [["A"], ["B"], ["C"]]


async def test_dependency_layers_multiple_dependencies():
    orchestrator = Orchestrator(goal="g", client=FakeLLMClient([]))
    tasks = [_task("A"), _task("B"), _task("D", ["A", "B"])]

    layers = await _layers_with_deadline(orchestrator, tasks)

    # D waits for both, so A and B stay co-scheduled in the first layer.
    assert _layer_ids(layers) == [["A", "B"], ["D"]]


async def test_dependency_layers_unknown_dependency_is_satisfied():
    orchestrator = Orchestrator(goal="g", client=FakeLLMClient([]))
    tasks = [_task("A", ["does-not-exist"]), _task("B")]

    layers = await _layers_with_deadline(orchestrator, tasks)

    assert _layer_ids(layers) == [["A", "B"]]


async def test_dependency_layers_self_dependency_does_not_hang():
    orchestrator = Orchestrator(goal="g", client=FakeLLMClient([]))
    tasks = [_task("A", ["A"])]

    layers = await _layers_with_deadline(orchestrator, tasks)

    assert _layer_ids(layers) == [["A"]]


async def test_dependency_layers_cycle_does_not_hang():
    orchestrator = Orchestrator(goal="g", client=FakeLLMClient([]))
    tasks = [_task("A", ["B"]), _task("B", ["A"])]

    layers = await _layers_with_deadline(orchestrator, tasks)

    flat = [task.id for layer in layers for task in layer]
    assert sorted(flat) == ["A", "B"]
    assert len(flat) == 2, "each task must be emitted exactly once"


async def test_dependency_layers_empty_input():
    orchestrator = Orchestrator(goal="g", client=FakeLLMClient([]))

    layers = await _layers_with_deadline(orchestrator, [])

    assert layers == []


async def test_dependency_layers_preserves_input_order():
    orchestrator = Orchestrator(goal="g", client=FakeLLMClient([]))
    tasks = [_task("gamma"), _task("alpha"), _task("beta")]

    layers = await _layers_with_deadline(orchestrator, tasks)

    assert _layer_ids(layers) == [["gamma", "alpha", "beta"]]


async def test_dependent_task_waits_for_its_dependency(timeline_agents):
    # Regression guard: a flat `asyncio.gather` would start A and B together.
    client = FakeLLMClient([dependency_plan({"A": [], "B": ["A"]})])
    orchestrator = Orchestrator(goal="build it", client=client)

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    events = timeline_agents.events
    assert events, "no worker ever ran"
    started = [task_id for kind, task_id in events if kind == "start"]
    assert started == ["A", "B"]
    assert events.index(("start", "A")) < events.index(("finish", "A"))
    assert events.index(("finish", "A")) < events.index(("start", "B"))
    assert [task.status for task in orchestrator.all_tasks] == ["completed", "completed"]


async def test_independent_tasks_still_run_in_parallel(agents):
    agents.worker_delay = 0.02
    client = FakeLLMClient([plan_payload(3)])
    orchestrator = Orchestrator(goal="build it", client=client)

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    # Layering must not needlessly serialize dependency-free work.
    assert agents.peak > 1
    assert agents.peak <= client.settings.max_parallel_tasks
    assert all(task.status == "completed" for task in orchestrator.all_tasks)


async def test_parallelism_cap_still_applies_per_layer(agents):
    agents.worker_delay = 0.02
    plan = dependency_plan({"1": [], "2": [], "3": [], "4": [], "5": []})
    client = FakeLLMClient([plan], max_parallel_tasks=2)
    orchestrator = Orchestrator(goal="build it", client=client)

    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    # All five tasks sit in a single layer, so only the semaphore throttles them.
    assert len(orchestrator.all_tasks) == 5
    assert all(task.status == "completed" for task in orchestrator.all_tasks)
    assert agents.peak >= 1
    assert agents.peak <= 2


async def test_dependency_cycle_still_completes_tasks(agents):
    client = FakeLLMClient([dependency_plan({"A": ["B"], "B": ["A"]})])
    orchestrator = Orchestrator(goal="build it", client=client)

    # The workers never fail, so a terminated run leaves both tasks completed.
    await asyncio.wait_for(orchestrator._cycle(), timeout=TIMEOUT)

    assert {task.id: task.status for task in orchestrator.all_tasks} == {
        "A": "completed",
        "B": "completed",
    }
    assert orchestrator._is_done() is True