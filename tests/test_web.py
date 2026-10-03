"""Web-layer regression tests for :mod:`furrow.web.server`.

What is protected here:

* ``/api/health`` and ``/api/state`` answer sanely before anything has run, and
  the reported version falls back when the distribution metadata is missing.
* ``POST /api/run`` refuses an empty or whitespace-only goal with 422 and starts
  nothing -- a blank goal must never be able to spin the loop forever.
* Only one run exists at a time: a second start is 409, and ``/api/stop`` is 404
  while idle and 200 while running, reaching the live orchestrator.
* ``/ws`` streams event frames that end in a terminal frame, and a blank goal on
  the socket yields an ``error`` frame with no run started.
* Hanging up mid-run releases the run registry, awaits the detached task, and
  leaves no "Task was destroyed but it is pending" warning behind.
* The single page console derives its websocket scheme from ``location.protocol``
  and never writes server supplied text as HTML.

Everything is offline: :class:`furrow.core.Orchestrator` is replaced by a scripted
double, so no LLM, subprocess or workspace is touched, and every read carries a
timeout so a hang fails fast instead of wedging the suite.
"""

from __future__ import annotations

import asyncio
import gc
import inspect
import json
import logging
import threading
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from typing import Any

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from furrow.core import Event, EventType
from furrow.web import server
from furrow.web.server import _empty_state

#: A websocket frame that must arrive within this many seconds, or the test fails.
FRAME_TIMEOUT = 5.0
#: A detached run task must finish within this many seconds, or the test fails.
RUN_TIMEOUT = 5.0
#: Upper bound on frames read from one socket, so a chatty server cannot loop forever.
FRAME_LIMIT = 32
#: Frame types that end a websocket stream.
TERMINAL_FRAMES = frozenset({"done", "error"})


# --------------------------------------------------------------------- doubles


class ScriptedOrchestrator:
    """Offline stand-in for :class:`furrow.core.Orchestrator`.

    It records what the server asked of it and replays a fixed event script, so a
    run finishes in microseconds and nothing is executed. With ``hold`` set the
    run parks after its first event until :meth:`request_stop` arrives, which is
    how the "a run is in progress" window is held open deterministically.
    """

    def __init__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        on_event: Callable[[Event], Any] | None = None,
        *,
        hold: bool = False,
    ) -> None:
        self.goal = goal
        self.client = client
        self.settings = settings
        self.on_event = on_event
        self.hold = hold
        self.cycles = 0
        self.done = False
        self.stop_requested = False
        self.run_calls: list[int | None] = []
        self._released = asyncio.Event()

    # ------------------------------------------------------------ orchestrator API

    def request_stop(self) -> None:
        """Record the stop request and let a held run unwind."""
        self.stop_requested = True
        self._released.set()

    def state(self) -> dict[str, Any]:
        """Return the same JSON-serialisable snapshot shape as the real loop."""
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "done": self.done,
            "tasks": [
                {
                    "id": "1",
                    "description": "do the thing",
                    "status": "completed",
                    "files": [],
                    "dependencies": [],
                    "result": "ok",
                }
            ],
            "history": [],
        }

    async def run(self, max_cycles: int | None = None) -> None:
        """Replay one scripted cycle, optionally parking until stopped first."""
        self.run_calls.append(max_cycles)
        await self._emit(EventType.CYCLE_START, "cycle 1 started")
        if self.hold:
            await self._released.wait()
        self.cycles = 1
        task = self.state()["tasks"][0]
        await self._emit(EventType.PLAN, "one task planned", {"tasks": task})
        await self._emit(EventType.TASK_START, task["description"], {"task": task})
        await self._emit(EventType.TASK_END, task["description"], {"task": task})
        await self._emit(EventType.TEST_START, "running tests", {"goal": self.goal})
        await self._emit(EventType.TEST_END, "1 passed", {"passed": True, "summary": "1 passed"})
        await self._emit(EventType.CYCLE_END, "cycle 1 finished")
        self.done = True
        await self._emit(
            EventType.DONE,
            "Goal complete.",
            {"goal": self.goal, "cycles": self.cycles, "done": True},
        )

    async def _emit(self, event_type: EventType, message: str, data: Any = None) -> None:
        """Hand one event to the server's callback, awaiting it when needed."""
        if self.on_event is None:
            return
        outcome = self.on_event(
            Event(type=event_type, cycle=self.cycles, message=message, data=data or {})
        )
        if inspect.isawaitable(outcome):
            await outcome


class ScriptedFactory:
    """Callable installed as ``server.Orchestrator``; collects every double built."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.created: list[ScriptedOrchestrator] = []
        self.hold = False
        monkeypatch.setattr(server, "Orchestrator", self)

    def __call__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        on_event: Callable[[Event], Any] | None = None,
    ) -> ScriptedOrchestrator:
        """Build one scripted double for the server."""
        orchestrator = ScriptedOrchestrator(
            goal, client=client, settings=settings, on_event=on_event, hold=self.hold
        )
        self.created.append(orchestrator)
        return orchestrator

    @property
    def only(self) -> ScriptedOrchestrator:
        """Return the single double that was built, failing otherwise."""
        assert len(self.created) == 1, f"expected exactly one run, built {len(self.created)}"
        return self.created[0]


# --------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def reset_server_globals(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Give every test a pristine run registry.

    ``_active_run``, ``_last_state`` and ``_background_tasks`` are process-wide
    module globals, so without this reset one test's run would leak into the next
    and the suite would depend on execution order.
    """
    monkeypatch.setattr(server, "_active_run", None)
    monkeypatch.setattr(server, "_last_state", _empty_state())
    monkeypatch.setattr(server, "_background_tasks", set())
    yield
    # Teardown runs after the ``client`` fixture has drained the portal loop.
    assert server._active_run is None, "a run leaked out of the process-wide registry"
    assert not server._background_tasks, "a detached run task was never awaited"


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedFactory:
    """Replace the orchestrator with offline scripted doubles."""
    return ScriptedFactory(monkeypatch)


@pytest.fixture
def client() -> Any:
    """A ``TestClient`` whose portal keeps one background run alive across requests."""
    with TestClient(server.app) as test_client:
        yield test_client
        # Drain before the portal's event loop closes: a task still pending here
        # would be destroyed with the loop and asyncio would log a warning.
        test_client.portal.call(_stop_and_drain, RUN_TIMEOUT)


# ---------------------------------------------------------------- sync helpers


async def _await_runs(timeout: float) -> None:
    """Wait for every detached run task to finish, or fail loudly instead of hanging."""
    tasks = list(server._background_tasks)
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=timeout)
    assert not pending, f"run tasks still pending after {timeout}s: {pending!r}"


async def _stop_and_drain(timeout: float) -> None:
    """Stop the active run, cancel leftover run tasks and await them."""
    run = server._active_run
    if run is not None:
        run.orchestrator.request_stop()
    for task in list(server._background_tasks):
        task.cancel()
    await _await_runs(timeout)
    server._background_tasks.clear()


def receive_json(socket: Any, timeout: float = FRAME_TIMEOUT) -> dict[str, Any]:
    """Read one websocket frame under a hard timeout.

    ``WebSocketTestSession.receive`` blocks the calling thread forever when the
    server never writes, so the read is bounded here and reported as a test
    failure rather than a wedged suite.
    """
    outcome: dict[str, Any] = {}

    def _read() -> None:
        try:
            outcome["frame"] = socket.receive_json()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller thread
            outcome["error"] = exc

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    reader.join(timeout)
    if reader.is_alive():
        pytest.fail(f"no websocket frame arrived within {timeout}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["frame"]


def collect_frames(socket: Any, limit: int = FRAME_LIMIT) -> list[dict[str, Any]]:
    """Read frames until a terminal one arrives or the server closes the socket."""
    frames: list[dict[str, Any]] = []
    for _ in range(limit):
        try:
            frame = receive_json(socket)
        except (WebSocketDisconnect, json.JSONDecodeError):
            break
        frames.append(frame)
        if frame.get("type") in TERMINAL_FRAMES:
            break
    return frames


# --------------------------------------------------------------------- REST API


def test_health_reports_ok_and_a_version(client: TestClient) -> None:
    """``/api/health`` is 200 with a status, and the version falls back off-distribution."""
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == server._package_version()
    try:
        installed = metadata_version("furrow")
    except PackageNotFoundError:
        # The distribution really is not installed here, so the fallback is live code.
        assert body["version"] == "0.1.0"
    else:
        assert body["version"] == installed


def test_state_before_any_run_is_serialisable_and_empty(client: TestClient) -> None:
    """``/api/state`` is 200 and reports the documented idle snapshot."""
    response = client.get("/api/state")
    assert response.status_code == 200
    body = response.json()
    assert body == _empty_state()
    assert set(body) == {"goal", "cycles", "done", "tasks", "history"}
    assert body["goal"] is None
    assert body["cycles"] == 0
    assert body["done"] is False
    assert json.loads(json.dumps(body)) == body


def test_state_tracks_the_run_that_is_in_progress(
    client: TestClient, scripted: ScriptedFactory
) -> None:
    """While a run holds the registry, ``/api/state`` serves that run's snapshot."""
    scripted.hold = True
    assert client.post("/api/run", json={"goal": "watch me"}).status_code == 202

    body = client.get("/api/state").json()

    assert body["goal"] == "watch me"
    assert body["cycles"] == 0
    assert body["done"] is False


@pytest.mark.parametrize("goal", ["", "   ", "\t\n  "])
def test_run_rejects_a_blank_goal_with_422(
    client: TestClient, scripted: ScriptedFactory, goal: str
) -> None:
    """A blank goal must be refused outright: it must never start the loop."""
    response = client.post("/api/run", json={"goal": goal})

    assert response.status_code == 422
    assert scripted.created == []
    assert server._active_run is None
    assert not server._background_tasks


def test_run_accepts_a_goal_and_echoes_it_trimmed(
    client: TestClient, scripted: ScriptedFactory
) -> None:
    """A valid goal starts a run in the background and echoes the trimmed goal."""
    response = client.post("/api/run", json={"goal": "  ship the console  "})

    assert response.status_code == 202
    assert response.json() == {"started": True, "goal": "ship the console", "max_cycles": None}
    client.portal.call(_await_runs, RUN_TIMEOUT)
    assert scripted.only.goal == "ship the console"
    # No cycle budget was supplied, so the run is bounded by the settings default.
    assert scripted.only.run_calls == [None]


def test_a_second_concurrent_run_is_rejected_with_409(
    client: TestClient, scripted: ScriptedFactory
) -> None:
    """Only one run may exist at a time; a second start is refused, not queued."""
    scripted.hold = True
    first = client.post("/api/run", json={"goal": "first goal"})
    second = client.post("/api/run", json={"goal": "second goal"})

    assert first.status_code == 202
    assert second.status_code == 409
    assert "first goal" in second.json()["detail"]
    assert len(scripted.created) == 1
    assert scripted.only.goal == "first goal"


def test_stop_while_idle_is_404(client: TestClient) -> None:
    """Stopping when nothing is running is a 404, not a crash."""
    response = client.post("/api/stop")

    assert response.status_code == 404
    assert server._active_run is None


def test_stop_while_running_reaches_the_orchestrator_and_ends_the_run(
    client: TestClient, scripted: ScriptedFactory
) -> None:
    """``/api/stop`` reaches the live orchestrator and the run really finishes."""
    scripted.hold = True
    assert client.post("/api/run", json={"goal": "stoppable"}).status_code == 202
    orchestrator = scripted.only
    assert orchestrator.stop_requested is False

    response = client.post("/api/stop")

    assert response.status_code == 200
    assert response.json() == {"stopped": True}
    assert orchestrator.stop_requested is True

    client.portal.call(_await_runs, RUN_TIMEOUT)
    assert server._active_run is None
    final = client.get("/api/state").json()
    assert final["goal"] == "stoppable"
    assert final["done"] is True


# ------------------------------------------------------------------- websocket


def test_websocket_streams_events_and_ends_with_done(
    client: TestClient, scripted: ScriptedFactory
) -> None:
    """A valid goal streams event frames and the stream ends with a terminal ``done``."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "stream me"})
        frames = collect_frames(socket)

    types = [frame["type"] for frame in frames]
    assert types[0] == "cycle_start"
    assert types[-1] == "done"
    assert "task_start" in types and "test_end" in types
    assert "error" not in types
    assert scripted.only.goal == "stream me"
    assert server._active_run is None
    assert not server._background_tasks


def test_websocket_rejects_a_blank_goal_with_an_error_frame(
    client: TestClient, scripted: ScriptedFactory
) -> None:
    """A blank first frame is reported as an ``error`` frame and starts nothing."""
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": ""})
        frames = collect_frames(socket)

    assert [frame["type"] for frame in frames] == ["error"]
    assert "goal" in frames[0]["message"]
    assert scripted.created == []
    assert server._active_run is None
    assert not server._background_tasks


def test_disconnecting_mid_run_tears_everything_down(
    client: TestClient, scripted: ScriptedFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """Hanging up mid-run releases the registry, awaits the task and logs no warning."""
    scripted.hold = True
    gc.collect()
    caplog.clear()

    with caplog.at_level(logging.ERROR, logger="asyncio"):
        started = time.monotonic()
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"goal": "stream me"})
            # The first frame only arrives once the run is really under way.
            assert receive_json(socket)["type"] == "cycle_start"
        elapsed = time.monotonic() - started

    gc.collect()
    orchestrator = scripted.only
    assert elapsed < 5.0, "closing the socket did not return promptly"
    assert orchestrator.stop_requested is True
    assert server._active_run is None
    assert not server._background_tasks
    assert "Task was destroyed but it is pending" not in caplog.text


# --------------------------------------------------------------------- console


def test_index_serves_the_console_html(client: TestClient) -> None:
    """``/`` serves the single page console as HTML."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Furrow console</title>" in response.text


def test_index_derives_the_websocket_scheme_from_the_page_protocol(client: TestClient) -> None:
    """The page builds its socket URL from ``location.protocol`` so TLS stays TLS."""
    html = client.get("/").text

    assert "location.protocol" in html
    assert "'https:'" in html
    assert "'wss'" in html
    assert "'ws'" in html


def test_index_writes_server_text_only_through_textcontent(client: TestClient) -> None:
    """Server supplied strings are never parsed as HTML (no ``innerHTML`` anywhere)."""
    html = client.get("/").text

    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert forbidden not in html
    assert "item.textContent" in html
    assert "statusLine.textContent" in html
    assert "taskLine.textContent" in html