"""Offline end-to-end test for the REAL orchestrator + REAL web server.

tests/test_web.py stubs ``Orchestrator`` wholesale and tests/test_orchestrator.py
never touches ``furrow.web.server``, so neither can catch a contract drift
between the producer (furrow/core/orchestrator.py) and the consumer
(furrow/web/server.py). This module wires the two REAL modules together.

Only the LLM boundary is faked: ``PlannerAgent``, ``TesterAgent``,
``WorkerAgent`` and ``LLMClient`` are replaced on the orchestrator module. Every
line of furrow/ that produces or consumes events runs for real, and no test
here can reach the network -- the fake client raises on any call.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.config import settings as config_singleton
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator
from furrow.web import server as server_module
from furrow.web.server import make_on_event, settings_for

# Hard cap on every blocking interaction, so a contract regression fails the
# suite instead of hanging it.
TIMEOUT = 10.0

WS = Path(".")

# The frozen cross-component contract.
ALLOWED_EVENT_TYPES = frozenset(
    {"cycle_start", "plan", "task", "test", "cycle_end", "done"}
)
CYCLE_SCOPED_TYPES = ALLOWED_EVENT_TYPES - {"done"}

PASSING = TestResult(passed=True, summary="all green")
FAILING = TestResult(
    passed=False,
    summary="1 failed",
    failures=["tests/test_x.py::test_y asserts 2 == 3"],
)


class FakeLLM:
    """Offline stand-in for LLMClient; any real call is a test bug."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        raise AssertionError("FakeLLM.complete must never be called")

    async def read_file(self, path: str) -> str:
        raise AssertionError("FakeLLM.read_file must never be called")

    async def write_file(self, path: str, content: str) -> None:
        raise AssertionError("FakeLLM.write_file must never be called")

    def list_files(self, directory: str) -> list[str]:
        return []


class RecordingSocket:
    """Duck-typed WebSocket: captures exactly what make_on_event sends."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


def make_plan(*descriptions: str) -> Plan:
    return Plan(
        tasks=[
            TaskModel(id=f"t{i}", description=desc)
            for i, desc in enumerate(descriptions)
        ],
        rationale="stub plan",
    )


def make_planner(plans: list[Plan], calls: list[str]) -> Any:
    """Class standing in for PlannerAgent (constructed with client=... only)."""

    class _Planner:
        def __init__(self, client: Any = None, settings: Any = None) -> None:
            self.client = client
            self.settings = settings

        async def plan(self, goal: str) -> Plan:
            calls.append(goal)
            if len(plans) > 1:
                return plans.pop(0)
            return plans[0]

    return _Planner


def make_tester(results: list[TestResult], calls: list[str]) -> Any:
    """Class standing in for TesterAgent."""

    class _Tester:
        def __init__(self, client: Any = None, settings: Any = None) -> None:
            self.client = client
            self.settings = settings

        async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
            calls.append(goal)
            if len(results) > 1:
                return results.pop(0)
            return results[0]

    return _Tester


def make_worker(completed: list[str]) -> Any:
    """Class standing in for WorkerAgent (constructed with task=/client=)."""

    class _Worker:
        def __init__(self, task: TaskModel, client: Any = None) -> None:
            self.task = task
            self.client = client

        async def run(self) -> str:
            await asyncio.sleep(0)
            completed.append(self.task.id)
            return f"done:{self.task.id}"

    return _Worker


def build_real_orchestrator(
    goal: str = "ship the feature",
    plans: list[Plan] | None = None,
    results: list[TestResult] | None = None,
    settings: Settings | None = None,
    on_event: Any = None,
    completed: list[str] | None = None,
) -> tuple[Orchestrator, list[str], list[str]]:
    """Real Orchestrator with only the LLM/agent boundary stubbed."""
    planner_calls: list[str] = []
    tester_calls: list[str] = []
    done_tasks: list[str] = [] if completed is None else completed

    orch = Orchestrator(
        goal=goal,
        client=FakeLLM(),
        settings=settings if settings is not None else Settings(max_cycles=10, workspace=WS),
        on_event=on_event,
    )
    # The factories return classes; the orchestrator holds instances.
    orch.planner = make_planner(plans or [make_plan("do a thing")], planner_calls)()
    tester_results = [PASSING] if results is None else results
    orch.tester = make_tester(tester_results, tester_calls)()
    # Replace only the worker seam; _guarded and the gather stay real.
    orch._run_worker = _make_run_worker(done_tasks)
    return orch, planner_calls, tester_calls


def _make_run_worker(completed: list[str]) -> Any:
    async def _run(task: TaskModel) -> str:
        await asyncio.sleep(0)
        completed.append(task.id)
        return f"done:{task.id}"

    return _run


async def _run_bounded(orch: Orchestrator) -> None:
    await asyncio.wait_for(orch.run(), timeout=TIMEOUT)


def assert_frozen_contract(events: list[dict[str, Any]]) -> None:
    """Everything furrow/web/server.py may assume about a real event."""
    assert events, "no events were captured"
    for event in events:
        assert set(event) >= {"type"}, f"event missing type: {event!r}"
        kind = event["type"]
        assert kind in ALLOWED_EVENT_TYPES, f"undocumented event type {kind!r}"
        assert isinstance(event, dict)
        if kind == "done":
            assert "reason" in event, f"done event carries no reason: {event!r}"
            assert "cycle" not in event, f"done event must not carry cycle: {event!r}"
        else:
            assert kind in CYCLE_SCOPED_TYPES
            assert type(event["cycle"]) is int, f"cycle must be an int: {event!r}"


async def test_real_orchestrator_events_survive_the_real_web_encoder() -> None:
    """Producer/consumer interop: real _emit -> real make_on_event -> json."""
    socket = RecordingSocket()
    events: list[dict[str, Any]] = []
    orch, planner_calls, _ = build_real_orchestrator(
        goal="ship the feature",
        plans=[make_plan("write it"), make_plan("write it")],
        results=[FAILING, PASSING],
        settings=Settings(max_cycles=10, max_parallel_tasks=2, workspace=WS),
    )

    # The REAL callback built by the REAL web module, wrapped so the raw dicts
    # stay available for structural assertions.
    send = make_on_event(socket)

    async def on_event(event: dict[str, Any]) -> None:
        events.append(event)
        await send(event)

    orch.on_event = on_event
    await _run_bounded(orch)

    assert_frozen_contract(events)
    # make_on_event terminates every frame, and every frame is parseable JSON
    # carrying exactly the dict the orchestrator emitted.
    assert len(socket.sent) == len(events)
    for raw, event in zip(socket.sent, events, strict=True):
        assert raw.endswith("\n"), f"unterminated websocket frame: {raw!r}"
        assert json.loads(raw) == event

    # A failing cycle followed by a passing one, then done.
    assert [e["cycle"] for e in events if e["type"] == "cycle_start"] == [1, 2]
    assert [e["result"]["passed"] for e in events if e["type"] == "test"] == [
        False,
        True,
    ]
    assert events[-1] == {"type": "done", "reason": "complete"}
    assert [e["cycle"] for e in events if e["type"] == "cycle_end"] == [1, 2]
    assert orch.cycles == 2

    # Goal is immutable; only focus carries the failure text, for one cycle.
    assert orch.goal == "ship the feature"
    assert {e["goal"] for e in events if "goal" in e} == {"ship the feature"}
    assert planner_calls[0] == "ship the feature"
    assert planner_calls[1].startswith("Fix failing tests:")
    assert "tests/test_x.py::test_y asserts 2 == 3" in planner_calls[1]


async def test_focus_carries_the_fix_text_while_goal_survives() -> None:
    """Assert the mid-run state, not just the post-run state."""
    events: list[dict[str, Any]] = []
    focus_at_cycle_end: dict[int, str] = {}
    orch, _, tester_calls = build_real_orchestrator(
        goal="never mutate me",
        plans=[make_plan("only"), make_plan("only")],
        results=[FAILING, PASSING],
        settings=Settings(max_cycles=10, workspace=WS),
    )

    def on_event(event: dict[str, Any]) -> None:
        events.append(event)
        if event["type"] == "cycle_end":
            focus_at_cycle_end[event["cycle"]] = orch.focus

    orch.on_event = on_event
    await _run_bounded(orch)

    assert focus_at_cycle_end[1].startswith("Fix failing tests:")
    assert "tests/test_x.py::test_y asserts 2 == 3" in focus_at_cycle_end[1]
    # The leak lasts exactly one cycle.
    assert focus_at_cycle_end[2] == "never mutate me"
    assert orch.goal == "never mutate me"
    assert orch.focus == "never mutate me"
    # The tester only ever sees the real goal, never the focus text.
    assert tester_calls == ["never mutate me", "never mutate me"]


async def test_max_cycles_is_honoured_by_the_real_loop() -> None:
    events: list[dict[str, Any]] = []
    orch, _, _ = build_real_orchestrator(
        plans=[make_plan("never finishes")],
        results=[FAILING],
        settings=Settings(max_cycles=2, workspace=WS),
    )
    orch.on_event = events.append
    await _run_bounded(orch)

    assert orch.cycles == 2
    assert [e["cycle"] for e in events if e["type"] == "cycle_start"] == [1, 2]
    assert events[-1] == {"type": "done", "reason": "max_cycles"}
    assert_frozen_contract(events)


async def test_max_parallel_tasks_is_honoured_by_the_real_guarded() -> None:
    active = 0
    peak = 0

    async def worker(task: TaskModel) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return f"done:{task.id}"

    events: list[dict[str, Any]] = []
    orch, _, _ = build_real_orchestrator(
        plans=[make_plan("a", "b", "c", "d")],
        results=[PASSING],
        settings=Settings(max_cycles=1, max_parallel_tasks=2, workspace=WS),
    )
    orch._run_worker = worker
    orch.on_event = events.append
    await _run_bounded(orch)

    assert peak == 2, f"observed concurrency {peak}, expected a cap of 2"
    assert [t.status for t in orch.tasks] == ["completed"] * 4
    assert len([e for e in events if e["type"] == "task"]) == 4
    assert_frozen_contract(events)


def test_orchestrator_accepts_the_frozen_constructor_keywords() -> None:
    params = inspect.signature(Orchestrator.__init__).parameters
    for name in ("goal", "client", "settings", "on_event"):
        assert name in params, f"Orchestrator.__init__ lost keyword {name!r}"
        assert params[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    # The web module calls it exactly like this, with no client.
    orch = Orchestrator(
        goal="x",
        client=FakeLLM(),
        settings=Settings(max_cycles=1, workspace=WS),
        on_event=None,
    )
    assert orch.goal == "x"
    assert orch.focus == "x"
    assert orch.on_event is None
    assert Orchestrator(goal="y").goal == "y"


def test_settings_for_override_copies_and_leaves_singleton_alone() -> None:
    assert server_module.settings is config_singleton

    before = config_singleton.model_dump()
    copied = settings_for("override-model")
    same = settings_for(None)

    assert copied is not config_singleton
    assert same is config_singleton
    assert copied.model == "override-model"
    # A copy, not an alias: mutating it cannot reach the singleton.
    copied.model = "mutated-locally"
    copied.max_parallel_tasks = 99
    assert config_singleton.model_dump() == before
    assert settings_for(None) is config_singleton
    assert config_singleton.model != "override-model"
    # Everything other than the override survives.
    overridden = settings_for("override-model")
    assert overridden.model == "override-model"
    assert overridden.max_parallel_tasks == config_singleton.max_parallel_tasks
    assert overridden.provider == config_singleton.provider
    # And the override is JSON-shaped, as the run path assumes.
    json.dumps(overridden.model_dump(), default=str)


async def test_real_orchestrator_streams_events_over_the_real_ws_socket() -> None:
    """The real /ws handler, the real Orchestrator, the real make_on_event."""
    import pytest  # local import keeps the module import graph flat

    planner_calls: list[str] = []
    tester_calls: list[str] = []
    completed: list[str] = []
    run_settings = Settings(max_cycles=1, max_parallel_tasks=2, workspace=WS)

    monkeypatch = pytest.MonkeyPatch()
    singleton_before = config_singleton.model_dump()
    monkeypatch.setattr(orchestrator_module, "LLMClient", FakeLLM)
    monkeypatch.setattr(
        orchestrator_module, "PlannerAgent", make_planner([make_plan("one", "two")], planner_calls)
    )
    monkeypatch.setattr(orchestrator_module, "TesterAgent", make_tester([PASSING], tester_calls))
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", make_worker(completed))
    monkeypatch.setattr(server_module, "settings", run_settings)
    client = TestClient(server_module.app)
    received: list[dict[str, Any]] = []

    try:

        def exercise() -> None:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"goal": "  ship it  "})
                # cycle_start, plan, task, task, test, cycle_end, done
                for _ in range(7):
                    text = ws.receive_text()
                    assert text.endswith("\n"), "frames must be newline terminated"
                    received.append(json.loads(text))

        await asyncio.wait_for(asyncio.to_thread(exercise), timeout=TIMEOUT)
    finally:
        client.close()
        monkeypatch.undo()

    assert_frozen_contract(received)
    assert [e["type"] for e in received] == [
        "cycle_start",
        "plan",
        "task",
        "task",
        "test",
        "cycle_end",
        "done",
    ]
    assert [e["cycle"] for e in received if e["type"] == "cycle_start"] == [1]
    # The handler strips the goal and it reaches the real orchestrator.
    assert received[0]["goal"] == "ship it"
    assert received[-1] == {"type": "done", "reason": "complete"}
    # The real loop ran end to end behind the socket.
    assert planner_calls == ["ship it"]
    assert tester_calls == ["ship it"]
    assert sorted(completed) == ["t0", "t1"]
    # The socket path never mutated the process-wide settings singleton.
    assert config_singleton.model_dump() == singleton_before