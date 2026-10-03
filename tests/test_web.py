"""Tests for the FastAPI front-end: the HTML console page and the ``/ws`` frame stream.

The websocket endpoint is the interesting part.  It used to accept a connection,
kick off an orchestrator run and then never call ``send``: the browser's ``onmessage``
never fired, so the page stayed blank while the agent worked and there was no way to
stop the run.  Every socket read below is therefore bounded by a *frame count* rather
than a timeout -- this environment has no ``pytest-timeout`` plugin, so an unbounded
receive would hang the entire suite instead of failing one test.

Every orchestrator is faked; no test here may reach an LLM provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from furrow.config import Settings
from furrow.web import server
from furrow.web.server import MAX_RUN_SECONDS, app

if TYPE_CHECKING:
    from starlette.testclient import WebSocketTestSession

#: Hard ceiling on frames read from a single socket.  A read loop stops here and
#: fails loudly, so an endpoint that streams forever cannot wedge the suite.
MAX_FRAMES = 50

#: Text the fake orchestrator renders; a ``log`` frame must carry it verbatim.
MARKER = "fake-orchestrator-output"

#: The page heading, as served.
PAGE_TITLE = "Furrow"


def _collect_frames(
    session: WebSocketTestSession, *, limit: int = MAX_FRAMES
) -> list[dict[str, Any]]:
    """Read JSON frames until the server closes the socket.

    Bounded by ``limit`` receives *and* by the server closing: the endpoint always
    closes after ``done``, so the loop normally exits on the first frame.  Hitting
    ``limit`` means the endpoint never hung up, which is a failure, not a wait.
    """
    frames: list[dict[str, Any]] = []
    for _ in range(limit):
        try:
            frame = session.receive_json()
        except WebSocketDisconnect:
            return frames
        assert isinstance(frame, dict), f"expected a JSON object frame, got {frame!r}"
        frames.append(frame)
    raise AssertionError(f"socket still open after {limit} frames: {frames!r}")


def _exchange(payload: Any) -> list[dict[str, Any]]:
    """Open ``/ws``, send one payload, and return every frame until close."""
    with TestClient(app).websocket_connect("/ws") as session:
        if isinstance(payload, str):
            session.send_text(payload)
        else:
            session.send_json(payload)
        return _collect_frames(session)


class _FakeOrchestrator:
    """Offline stand-in for :class:`furrow.core.orchestrator.Orchestrator`.

    The endpoint constructs the orchestrator as ``Orchestrator(goal=..., console=...)``
    because ``_ACCEPTS_CONSOLE_KWARG`` is frozen at import time and is ``True``, so
    this signature must keep accepting ``console`` or the endpoint raises ``TypeError``
    before any frame is sent.  All output goes through the injected console, which is
    exactly the stream the endpoint taps into frames.
    """

    instances: ClassVar[list[_FakeOrchestrator]] = []

    def __init__(self, goal: str, console: Any = None) -> None:
        self.goal = goal
        self.console = console
        self.task: asyncio.Task[Any] | None = None
        self.completed = False
        self.cancelled = False
        self.instances.append(self)

    def _render(self) -> None:
        """Emit a cycle banner (the ``status`` frame trigger) plus the marker line."""
        assert self.console is not None, "endpoint must inject a console"
        self.console.print("Cycle 1")
        self.console.print(MARKER)

    async def run(self) -> None:
        """Produce some output and finish immediately."""
        self.task = asyncio.current_task()
        self._render()
        self.completed = True


class _BlockingOrchestrator(_FakeOrchestrator):
    """Fake whose :meth:`run` never returns on its own, so only a cancel ends it."""

    async def run(self) -> None:
        self.task = asyncio.current_task()
        self._render()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.fixture
def fake_orchestrator() -> type[_FakeOrchestrator]:
    """Install :class:`_FakeOrchestrator` as ``server.Orchestrator``.

    Mandatory for every ``/ws`` test: without it the endpoint would build a real
    orchestrator and attempt real LLM calls.
    """
    _FakeOrchestrator.instances.clear()
    return _FakeOrchestrator


@pytest.fixture
def pinned_settings(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings]
) -> Settings:
    """Pin the settings the endpoint reports, so the startup frame text is fixed."""
    settings = make_settings(model="test-model", max_cycles=2)
    monkeypatch.setattr(server, "settings", settings)
    return settings


# --------------------------------------------------------------------------- page


def test_index_page_returns_ok() -> None:
    """The root route must actually serve the console page."""
    response = TestClient(app).get("/")
    assert response.status_code == 200


def test_index_page_contains_the_title_and_goal_input() -> None:
    """The page needs a heading and the goal box the browser sends over ``/ws``."""
    body = TestClient(app).get("/").text
    assert PAGE_TITLE in body
    assert 'id="goal"' in body


def test_index_page_references_the_websocket_endpoint() -> None:
    """The page script must talk to ``/ws``; anything else is a silent dead page."""
    assert "/ws" in TestClient(app).get("/").text


def test_index_page_is_usable_markup() -> None:
    """Guard against a stub body: a real console needs a control and a script."""
    body = TestClient(app).get("/").text
    assert "<button" in body
    assert "<script" in body
    assert "<textarea" in body


# ------------------------------------------------------------------- /ws errors


def test_ws_blank_goal_returns_one_error_frame() -> None:
    """A blank goal must be answered immediately with a readable error.

    Regression guard: before the fix the socket was accepted, the run started and no
    frame was ever sent, so a typo in the goal looked like a hung connection with no
    feedback at all.
    """
    frames = _exchange({"goal": ""})
    assert frames == [{"type": "error", "text": "goal must not be blank"}]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"goal": "   "}, id="whitespace-goal"),
        pytest.param({}, id="missing-goal-key"),
        pytest.param({"goal": None}, id="null-goal"),
    ],
)
def test_ws_rejects_unusable_goals_with_an_error_frame(payload: dict[str, Any]) -> None:
    """Malformed payloads get an ``error`` frame naming the offending field."""
    frames = _exchange(payload)
    assert len(frames) == 1, f"expected exactly one frame, got {frames!r}"
    assert frames[0]["type"] == "error"
    text = frames[0]["text"]
    assert isinstance(text, str) and text.strip(), f"error text must be readable: {text!r}"
    assert "goal" in text.lower(), f"error text should point at the goal field: {text!r}"


def test_ws_non_json_payload_returns_an_error_frame() -> None:
    """A client that sends garbage gets a diagnostic frame, not a silent close."""
    frames = _exchange("this is not json")
    assert len(frames) == 1, f"expected exactly one frame, got {frames!r}"
    assert frames[0]["type"] == "error"
    text = frames[0]["text"]
    assert isinstance(text, str) and text.strip()
    assert "invalid json" in text.lower(), f"error should say the payload was unparseable: {text!r}"


# ----------------------------------------------------------------- /ws streaming


def test_ws_streams_output_and_finishes_with_done(
    monkeypatch: pytest.MonkeyPatch,
    fake_orchestrator: type[_FakeOrchestrator],
    pinned_settings: Settings,
) -> None:
    """The run must stream frames as output is produced and end with one ``done``.

    This is the regression: ``/ws`` never called ``send`` at all, so the browser's
    ``onmessage`` never fired and the page showed nothing while the agent worked.
    """
    monkeypatch.setattr(server, "Orchestrator", fake_orchestrator)
    frames = _exchange({"goal": "ship a thing"})
    kinds = [frame["type"] for frame in frames]

    assert kinds[0] == "log", f"run should open with a log frame, got {kinds!r}"
    startup = frames[0]["text"]
    assert "starting run" in startup
    assert f"model={pinned_settings.model}" in startup
    assert f"cap={MAX_RUN_SECONDS}s" in startup

    assert kinds.count("status") == 1, f"expected one cycle status frame, got {kinds!r}"
    status_at = kinds.index("status")
    assert frames[status_at] == {"type": "status", "cycle": 1}
    assert status_at < kinds.index("done"), "status must precede done"
    assert "Cycle 1" in frames[status_at + 1]["text"], "cycle banner follows its status frame"

    logs = [frame["text"] for frame in frames if frame["type"] == "log"]
    assert any(MARKER in text for text in logs), f"captured output missing from {logs!r}"

    assert kinds[-1] == "done", f"done must be the terminal frame, got {kinds!r}"
    assert kinds.count("done") == 1
    assert frames[-1]["status"] == "completed"
    assert frames[-1]["reason"] == "run finished"

    run = fake_orchestrator.instances[-1]
    assert run.goal == "ship a thing"


def test_ws_does_not_leave_the_run_task_dangling(
    monkeypatch: pytest.MonkeyPatch,
    fake_orchestrator: type[_FakeOrchestrator],
    pinned_settings: Settings,
) -> None:
    """The endpoint owns the run task: it must finish or be cancelled, never leak."""
    monkeypatch.setattr(server, "Orchestrator", fake_orchestrator)
    _exchange({"goal": "ship a thing"})

    run = fake_orchestrator.instances[-1]
    assert run.task is not None, "the orchestrator should have been run as a task"
    assert run.task.done(), "the run task must be settled once the socket closes"
    assert run.completed is True
    assert run.cancelled is False


# ----------------------------------------------------------------- /ws cancelling


def test_ws_cancel_stops_the_run_and_sends_done(
    monkeypatch: pytest.MonkeyPatch, pinned_settings: Settings
) -> None:
    """An inbound ``cancel`` must end the run with a ``done`` frame.

    The blocking fake never finishes by itself, so a ``done`` frame can only come
    from the cancel path -- which is what lets the page's Stop button work instead of
    leaving the connection open until the safety cap.
    """
    _BlockingOrchestrator.instances.clear()
    monkeypatch.setattr(server, "Orchestrator", _BlockingOrchestrator)
    with TestClient(app).websocket_connect("/ws") as session:
        session.send_json({"goal": "an endless job"})
        session.send_json({"type": "cancel"})
        frames = _collect_frames(session)

    done = frames[-1]
    assert done["type"] == "done", f"cancel must finish with done, got {frames!r}"
    assert done["status"] == "stopped"
    assert "cancel" in done["reason"].lower()

    run = _BlockingOrchestrator.instances[-1]
    assert run.goal == "an endless job"
    # The endpoint cancelled the run task and awaited it in its ``finally`` block;
    # a still-pending task here would be a leak for the lifetime of the process.
    assert run.task is not None, "the blocking orchestrator should have been started"
    assert run.task.done(), "the run task must not be left pending"
    assert run.cancelled is True