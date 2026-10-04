"""Tests for :mod:`furrow.web.server`.

Regression suite for the defect that shipped: ``/ws`` ran ``Orchestrator.run()``
and never called ``send_json``, so the browser only ever showed "Starting..." and
"Closed." while the agent burned tokens invisibly.  Every streaming test here
drives :func:`furrow.web.server.run_session` against a fake socket, so no real
socket, no network and no LLM is involved; the orchestrator seams
(``_run_task`` / ``_run_tests`` and the assignable ``planner``) are the ones
already established in ``tests/test_orchestrator.py``.

The second shipped defect was structural: the run was an unbounded loop with no
cancellation, so a disconnect left an agent running forever.  Those tests assert
the runner task is ``done()`` and ``cancelled()`` and that
:func:`pending_session_tasks` is empty afterwards.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from rich.console import Console

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from furrow.web.server import EventStream, app, pending_session_tasks, run_session

EventPayload = dict[str, Any]


def make_client(*, max_cycles: int = 0, max_parallel_tasks: int = 2) -> LLMClient:
    """A real LLMClient with hermetic settings (no LLM call is ever made)."""
    return LLMClient(
        Settings(
            _env_file=None,
            max_cycles=max_cycles,
            max_parallel_tasks=max_parallel_tasks,
            anthropic_api_key="unused",
            openai_api_key="unused",
            workspace=Path("/tmp"),
        )
    )


def make_plan(*task_ids: str) -> Plan:
    return Plan(
        tasks=[
            TaskModel(id=task_id, description=f"do {task_id}", files=[f"{task_id}.py"])
            for task_id in task_ids
        ],
        rationale="scripted",
    )


def failing(summary: str = "1 failed") -> TestResult:
    return TestResult(passed=False, summary=summary, failures=["tests/test_a.py::test_x"])


def passing() -> TestResult:
    return TestResult(passed=True, summary="2 passed", failures=[])


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the rich output (and its live-refresh machinery) off stdout."""
    monkeypatch.setattr(
        orchestrator_module, "console", Console(file=io.StringIO(), width=120)
    )


class ScriptedPlanner:
    """Stands in for ``PlannerAgent``; the script is finite on purpose."""

    def __init__(self, plans: list[Plan]) -> None:
        self.plans = list(plans)
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        if not self.plans:
            raise AssertionError(
                f"planner.plan() was called {len(self.goals) + 1} time(s), but the script "
                f"only has {len(self.goals)} plan(s): the orchestrator is over-looping"
            )
        self.goals.append(goal)
        return self.plans.pop(0)


class ScriptedOrchestrator(Orchestrator):
    """Overrides the ``_run_task`` / ``_run_tests`` seams; no LLM is used."""

    def __init__(
        self,
        *,
        goal: str,
        on_event: Any,
        client: LLMClient,
        plans: list[Plan] | None = None,
        test_results: list[TestResult] | None = None,
        run_task: Any = None,
    ) -> None:
        super().__init__(goal=goal, client=client, on_event=on_event)
        self.scripted_planner = ScriptedPlanner(list(plans or []))
        self.planner = self.scripted_planner
        self._test_results = list(test_results or [])
        self._run_task_impl = run_task
        self.task_calls: list[TaskModel] = []
        self.run_task_ref: asyncio.Task[None] | None = None

    async def run(self) -> None:
        self.run_task_ref = asyncio.current_task()
        await super().run()

    async def _run_task(self, task: TaskModel) -> str:
        self.task_calls.append(task)
        if self._run_task_impl is not None:
            return await self._run_task_impl(task)
        return f"done: {task.id}"

    async def _run_tests(self, plan: Plan) -> TestResult:
        if not self._test_results:
            raise AssertionError("_run_tests() was called more times than the script allows")
        return self._test_results.pop(0)


class FakeWebSocket:
    """Implements the ``WebSocketLike`` slice the server uses.

    ``sent`` is what the server wrote; ``inbox`` is what the client sends back.
    """

    def __init__(self) -> None:
        self.sent: list[EventPayload] = []
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False
        self.break_after: int | None = None

    async def send_json(self, data: Any) -> None:
        if self.closed or (
            self.break_after is not None and len(self.sent) >= self.break_after
        ):
            raise RuntimeError("Cannot call send once a close message has been sent.")
        self.sent.append(data)

    async def receive_json(self) -> Any:
        message = await self.inbox.get()
        if isinstance(message, BaseException):
            raise message
        return message

    async def close(self, code: int = 1000) -> None:
        self.closed = True

    # -- test driver helpers ------------------------------------------------
    def client_says(self, message: Any) -> None:
        self.inbox.put_nowait(message)

    def client_disconnects(self, code: int = 1000) -> None:
        self.inbox.put_nowait(WebSocketDisconnect(code))

    def events(self, name: str) -> list[EventPayload]:
        return [payload for payload in self.sent if payload.get("event") == name]

    def names(self) -> list[str]:
        return [str(payload.get("event")) for payload in self.sent]


class Session:
    """Runs one ``run_session`` against a fake socket and a scripted run."""

    def __init__(self, **orchestrator_kwargs: Any) -> None:
        self.kwargs = orchestrator_kwargs
        self.socket = FakeWebSocket()
        self.orchestrator: ScriptedOrchestrator | None = None

    def factory(self, goal: str, on_event: Any) -> ScriptedOrchestrator:
        """Consumes the sink the server hands us, exactly like a real session."""
        self.orchestrator = ScriptedOrchestrator(
            goal=goal, on_event=on_event, **self.kwargs
        )
        return self.orchestrator

    async def run(self) -> str:
        return await run_session(self.socket, "ship it", make_orchestrator=self.factory)

    def events(self, name: str) -> list[EventPayload]:
        return self.socket.events(name)


async def _session_after_first_task(**kwargs: Any) -> tuple[asyncio.Task[str], Session]:
    """A session parked inside ``_run_task``, with ``started`` already set.

    Returns the task driving ``run_session`` and the session, so a test can
    deliver a stop message or a disconnect only once the run is really underway.
    """
    started = asyncio.Event()
    gate = asyncio.Event()

    async def parked(task: TaskModel) -> str:
        started.set()
        await gate.wait()  # never opens; the test cancels the run instead
        return "unreachable"

    session = Session(
        run_task=parked,
        plans=[make_plan("t1")],
        test_results=[],
        **kwargs,
    )
    driver = asyncio.create_task(session.run())
    await asyncio.wait_for(started.wait(), timeout=2)
    return driver, session


async def test_events_reach_the_websocket_as_json() -> None:
    """The shipped bug: nothing was ever sent, so the UI stayed blank."""
    session = Session(
        client=make_client(max_cycles=1),
        plans=[make_plan("t1", "t2")],
        test_results=[passing()],
    )

    reason = await asyncio.wait_for(session.run(), timeout=2)

    assert reason == "finished"
    assert session.socket.names()[:2] == ["start", "cycle_start"]
    assert {
        "plan",
        "task_start",
        "task_complete",
        "test_result",
        "cycle_end",
        "halt",
        "session_end",
    } <= set(session.socket.names())
    # The halt arrives before the session-end marker, so the client can render it.
    assert session.socket.names().index("halt") < session.socket.names().index("session_end")

    cycle_start = session.events("cycle_start")[0]
    assert cycle_start["cycle"] == 1
    # Payloads are forwarded verbatim, rich markup included; the page escapes it.
    assert "[bold cyan]" in cycle_start["message"]
    assert "Cycle 1" in cycle_start["message"]
    assert session.events("halt")[0] == {
        "event": "halt",
        "message": "[bold green]Goal complete. Halting.[/bold green]",
        "reason": "goal_complete",
        "cycles": 1,
    }
    # Every payload survived a JSON round trip, i.e. send_json can serialise it.
    for payload in session.socket.sent:
        assert json.loads(json.dumps(payload)) == payload
    assert pending_session_tasks() == []


async def test_disconnect_mid_run_cancels_the_agent() -> None:
    driver, session = await _session_after_first_task(client=make_client(max_cycles=0))

    session.socket.client_disconnects()
    reason = await asyncio.wait_for(driver, timeout=2)

    assert reason == "client_disconnected"
    orchestrator = session.orchestrator
    assert orchestrator is not None
    assert orchestrator.run_task_ref is not None
    assert orchestrator.run_task_ref.done()
    assert orchestrator.run_task_ref.cancelled()
    assert session.events("session_end")[0]["reason"] == "client_disconnected"
    assert "halt" not in session.socket.names()
    assert pending_session_tasks() == []


async def test_stop_message_stops_the_run() -> None:
    driver, session = await _session_after_first_task(client=make_client(max_cycles=0))

    session.socket.client_says({"action": "stop"})
    reason = await asyncio.wait_for(driver, timeout=2)

    assert reason == "client_stopped"
    orchestrator = session.orchestrator
    assert orchestrator is not None
    assert orchestrator.run_task_ref is not None
    assert orchestrator.run_task_ref.cancelled()
    assert session.events("session_end")[0]["reason"] == "client_stopped"
    # Stopping mid-cycle is not a halt: the run was cut short.
    assert "halt" not in session.socket.names()
    assert pending_session_tasks() == []


async def test_max_cycles_bounds_a_session() -> None:
    """``Settings.max_cycles`` is the runaway guard for the unbounded loop."""
    max_cycles = 2
    session = Session(
        client=make_client(max_cycles=max_cycles),
        plans=[make_plan("t1"), make_plan("t1")],
        test_results=[failing("boom 1"), failing("boom 2")],
    )

    reason = await asyncio.wait_for(session.run(), timeout=2)

    assert reason == "finished"
    halts = session.events("halt")
    assert [payload["reason"] for payload in halts] == ["max_cycles"]
    assert halts[0]["max_cycles"] == max_cycles
    assert halts[0]["cycles"] == max_cycles
    assert len(session.events("cycle_start")) == max_cycles
    assert session.events("session_end")[0]["cycles"] == max_cycles
    assert session.events("error") == []
    assert pending_session_tasks() == []


async def test_a_failing_task_does_not_escape_the_handler() -> None:
    async def boom(task: TaskModel) -> str:
        raise RuntimeError("worker exploded")

    session = Session(
        client=make_client(max_cycles=1),
        plans=[make_plan("t1")],
        test_results=[failing()],
        run_task=boom,
    )

    reason = await asyncio.wait_for(session.run(), timeout=2)

    assert reason == "finished"
    failed = session.events("task_failed")
    assert [payload["task_id"] for payload in failed] == ["t1"]
    assert "worker exploded" in failed[0]["message"]
    assert session.events("error") == []
    assert session.events("halt")[0]["reason"] == "max_cycles"
    assert pending_session_tasks() == []


async def test_a_planner_that_raises_is_reported_not_raised() -> None:
    session = Session(
        client=make_client(max_cycles=1),
        plans=[],  # the scripted planner raises: the script allows no plans
        test_results=[],
    )

    reason = await asyncio.wait_for(session.run(), timeout=2)

    assert reason == "finished"
    errors = session.events("error")
    assert len(errors) == 1
    assert "over-looping" in errors[0]["message"]
    assert session.events("session_end")[0]["reason"] == "finished"
    assert pending_session_tasks() == []


async def test_a_broken_socket_does_not_raise() -> None:
    session = Session(
        client=make_client(max_cycles=1),
        plans=[make_plan("t1")],
        test_results=[passing()],
    )
    session.socket.break_after = 2  # the client stops accepting writes mid-run

    reason = await asyncio.wait_for(session.run(), timeout=2)

    assert reason == "finished"
    assert len(session.socket.sent) == 2
    assert pending_session_tasks() == []


async def test_a_full_buffer_drops_the_oldest_event_instead_of_blocking() -> None:
    socket = FakeWebSocket()
    stream = EventStream(socket, maxsize=8)
    for index in range(9):
        stream.emit("task_start", {"event": "task_start", "message": f"event {index}"})

    assert stream.dropped == 1
    stream.finish(reason="finished", cycles=1)
    await asyncio.wait_for(stream.pump(), timeout=2)

    messages = [str(payload["message"]) for payload in socket.sent]
    assert "event 0" not in messages  # the oldest event made way for the newest
    assert "event 8" in messages
    assert messages[-1].endswith("after 1 cycle(s).")
    assert socket.sent[-1]["dropped"] == 1


def test_index_page_streams_events() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "new WebSocket(" in html
    assert "socket.onmessage" in html
    assert "JSON.parse" in html
    assert "payload.event" in html
    assert "write(name, payload.message)" in html
    assert "payload.cycle" in html
    assert "{ action: 'stop' }" in html


def test_start_message_without_a_goal_is_rejected() -> None:
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        socket.send_json({"nope": True})
        payload = socket.receive_json()

    assert payload["event"] == "error"
    assert payload["reason"] == "bad_request"
    assert "goal" in payload["message"]


def test_malformed_start_message_is_rejected() -> None:
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        socket.send_text("not json at all")
        payload = socket.receive_json()

    assert payload["event"] == "error"
    assert "JSON" in payload["message"]
