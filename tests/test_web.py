"""Behavioural tests for :mod:`furrow.web.server`.

The web server is the only place where the loop's event stream is shown to a
human, and it was broken in exactly that way: ``websocket_endpoint`` built an
``asyncio.Queue``, subscribed an enqueue handler to the
:class:`~furrow.core.events.EventBus`, and then never called ``send_json`` at
all -- so every dashboard in the world could only ever display "Starting...".
Most of this file is therefore about what actually comes *out* of the socket:
:func:`test_websocket_streams_every_event_to_the_client` counts the frames, so
the original bug cannot come back unnoticed.

Three things make this file unusual, and each is pinned by a test of its own:

* **The token is read at import time.** ``_WEB_TOKEN`` is captured from
  ``os.environ`` when the module body executes, so setting
  ``FURROW_WEB_TOKEN`` from a test has no effect on the already-imported
  server. :func:`web_token` reloads the module with the variable set and
  restores both the variable and the module afterwards. It saves and restores
  ``os.environ`` *itself* rather than using ``monkeypatch`` on purpose:
  monkeypatch undoes its changes after a test's own teardown hooks have run, so
  a fixture-driven reload would read the token while the environment still
  held it and leave the server permanently authenticated for every later test.
  ``importlib.reload`` reuses the same module object, so patching
  :data:`SERVER` still works; only ``app`` is replaced, which is why the
  :class:`~starlette.testclient.TestClient` is built inside the block.

* **The module is addressed through :data:`sys.modules`.** This is the same
  trap ``tests/test_cli.py`` documents for ``furrow.cli.main``, where the
  package re-exports ``main`` and shadows the module. ``furrow/web/__init__.py``
  re-exports only ``app`` and ``run``, so ``server`` still resolves to the
  module -- but a patch through a package attribute is exactly the kind of
  silent no-op that keeps passing, so
  :func:`test_web_package_does_not_shadow_the_server_module` keeps the
  distinction honest and
  :func:`test_module_object_patching_replaces_the_orchestrator_symbol` proves
  the patch is live.

* **A frame read has no timeout.** ``WebSocketTestSession.receive_json()``
  blocks inside anyio's ``BlockingPortal.call``, which takes no ``timeout``
  kwarg, and ``pytest-timeout`` is not installed. A regression that stopped the
  server from sending would therefore *hang* this file rather than fail it.
  Every socket test below reads an exact frame count and asserts the last
  frame is the terminal ``run.completed``, so the hang is the failure signal;
  no test loops on an open-ended "read until the socket closes".

Everything here is offline and deterministic: the orchestrator is always a
stub, no API key is present, and an autouse fixture clears ``FURROW_*`` plus
both API keys and ``chdir``s into a temp directory so pydantic-settings cannot
pick up a stray ``.env`` (``/api/start`` builds ``Settings()`` without
``_env_file=None``).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import furrow.cli
import furrow.web.server
from furrow.core.events import (
    Event,
    EventBus,
    EventType,
    cycle_completed,
    cycle_started,
    plan_created,
    run_started,
    task_completed,
    task_failed,
    task_started,
    tests_completed,
)

#: The real ``furrow.web.server`` *module*. Reached through ``sys.modules`` on
#: purpose -- see the module docstring. Every patch below goes through this
#: object rather than through the ``furrow.web`` package.
SERVER: ModuleType = sys.modules["furrow.web.server"]

# ``tests_completed`` matches pytest's default ``test*`` collection pattern, so
# pytest would collect it as a test and then fail on its parameters (which are
# not fixtures). Marking it non-collectable silences that; pytest's
# ``PyCollector.isnosetest()`` checks exactly this attribute. The flag is set on
# the imported function object, not in furrow/core/events.py -- ``tests/conftest.py``
# does the same thing for ``furrow.config.TestResult``, and
# ``tests/test_events.py`` for this same factory.
tests_completed.__test__ = False

#: The environment variable the server reads *at import time*.
TOKEN_ENV = "FURROW_WEB_TOKEN"

#: A syntactically valid shared secret. Only ever compared, never sent anywhere.
SECRET_TOKEN = "s3cr3t-web-token"

#: Poll interval for :func:`_wait_for`. The app runs on its own event loop in
#: another thread, so the test thread can only observe its progress by polling.
POLL = 0.01

#: How long :func:`_wait_for` gives an observable condition before failing,
#: instead of hanging.
WAIT_TIMEOUT = 5.0

#: Model-generated text that would execute as script if the dashboard ever
#: injected it as HTML. It rides along in the streamed events so the round-trip
#: through JSON is proven to be verbatim -- the page's defence is ``textContent``
#: on the client, pinned separately by
#: :func:`test_dashboard_never_touches_innerhtml`.
HOSTILE_RESULT = "</script><script>alert('xss')</script>"

#: A burst shaped like a real run: the orchestrator's own lifecycle events, one
#: full cycle of work, then a failing second cycle. Built with the real
#: factories from :mod:`furrow.core.events` so the ``type``/``cycle``/``data``
#: shape on the wire is the shape the loop actually emits.
STREAM_EVENTS: tuple[Event, ...] = (
    run_started("demo", max_cycles=3),
    cycle_started(1),
    plan_created(
        {
            "tasks": [{"id": "1", "description": "write it", "files": ["a.py"]}],
            "rationale": "the goal needs it",
        },
        cycle=1,
    ),
    task_started("1", "write it", ["a.py"], cycle=1),
    task_completed("1", HOSTILE_RESULT, 0.25, cycle=1),
    task_failed("2", "boom", 0.5, cycle=2),
    tests_completed(False, "1 failed", ["task 2: boom"], cycle=2),
    cycle_completed(2, False),
)

#: What the stubbed orchestrator reports as its cycle count. The terminal frame
#: is built from ``orchestrator.cycles``, so this proves the server reads the
#: attribute rather than inventing a number.
STUB_CYCLES = 3

#: The exact number of frames a clean ``/ws`` run must deliver, and why:
#:
#: * 1 -- the server's own ``run_started``, sent directly with ``send_json``
#:   rather than through the queue;
#: * ``len(STREAM_EVENTS)`` -- everything the stub emitted on its
#:   :class:`~furrow.core.events.EventBus`, forwarded by the pump;
#: * 1 -- the terminal ``run_completed`` the pump's ``finally`` always sends.
#:
#: This ordering is deterministic: ``asyncio.create_task(_drive_run(...))`` is
#: followed *synchronously* by ``bus.subscribe(_enqueue)`` with no ``await``
#: between them, so the stub cannot emit before the subscriber exists, and the
#: stub finishes enqueueing before its task completes, so the pump cannot see
#: an empty queue with a finished task.
EXPECTED_FRAMES = 1 + len(STREAM_EVENTS) + 1


# ── helpers ────────────────────────────────────────────────────────────────


def _wait_for(predicate: Callable[[], bool], *, what: str) -> None:
    """Block the test thread until ``predicate()`` holds, or fail the test.

    The ASGI app runs on its own event loop in another thread (starlette's
    ``BlockingPortal``), so the only way to observe its progress is to poll.
    Bounded on purpose: a bug that never completes a run fails the test instead
    of hanging the suite.
    """
    deadline = time.monotonic() + WAIT_TIMEOUT
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL)
    pytest.fail(f"timed out after {WAIT_TIMEOUT}s waiting for {what}")


@contextlib.contextmanager
def web_token(value: str | None) -> Iterator[ModuleType]:
    """Bind :data:`TOKEN_ENV` to ``value`` for the block, and reload the server.

    ``FURROW_WEB_TOKEN`` is read into ``_WEB_TOKEN`` when the module executes,
    so the only way to change it is to re-execute the module. The environment
    variable is saved and restored here rather than through ``monkeypatch``
    because monkeypatch undoes its changes *after* the test's own teardown,
    which would leave the module reloaded with a token the environment no
    longer has -- permanently authenticated for every later test.

    Reloading reuses the same module object (so :data:`SERVER` stays valid and
    ``monkeypatch.setattr(SERVER, ...)`` keeps working) and replaces ``app``,
    which is why the caller must build its client *inside* the block. It also
    resets ``_active_runs``, which conveniently isolates the single-run tests.

    Yields the reloaded module.
    """
    previous = os.environ.get(TOKEN_ENV)
    if value is None:
        os.environ.pop(TOKEN_ENV, None)
    else:
        os.environ[TOKEN_ENV] = value
    try:
        yield importlib.reload(SERVER)
    finally:
        if previous is None:
            os.environ.pop(TOKEN_ENV, None)
        else:
            os.environ[TOKEN_ENV] = previous
        importlib.reload(SERVER)


def make_stub(
    *,
    events: Sequence[Event] = (),
    hold: bool = False,
    cycles: int = 0,
) -> tuple[type[Any], list[Any]]:
    """Build a throwaway stand-in for :class:`Orchestrator` and its instance log.

    The server reads ``orchestrator.cycles`` and ``orchestrator.stopped`` and
    calls ``run()``, ``aclose()`` and ``request_stop(reason)``, so the stub
    supplies that whole surface and nothing else. A *fresh class* per call means
    ``events``/``hold``/``cycles`` are closure constants and no behaviour can
    leak from one test into the next.

    ``events`` are emitted onto the bus the server hands in, in order.
    ``hold`` keeps ``run()`` in flight until :meth:`Stub.release` is called, or
    forever if it is not -- which is how a run is kept in the server's registry
    while the test inspects it. ``cycles`` is what ``.cycles`` reports, and is
    echoed in the terminal frame.

    Returns ``(StubClass, created_instances)``.
    """
    created: list[Any] = []

    class Stub:
        """Stand-in for the real orchestrator. No network, no workspace, no LLM."""

        def __init__(
            self,
            goal: str,
            *,
            bus: EventBus | None = None,
            settings: Any = None,
        ) -> None:
            self.goal = goal
            self.bus = bus
            self.settings = settings
            self.cycles = cycles
            self.stopped = False
            self.started = False
            self.finished = False
            self.cancelled = False
            self.closed = False
            self.stop_reasons: list[str] = []
            self._release_requested = False
            created.append(self)

        async def run(self) -> None:
            """Emit the configured events, then optionally stay in flight."""
            self.started = True
            if self.bus is not None:
                for event in events:
                    await self.bus.emit(event)
            if hold:
                try:
                    # A polled flag, not an asyncio.Event: release() is called
                    # from the test thread, and asyncio primitives are not
                    # thread-safe.
                    while not self._release_requested:
                        await asyncio.sleep(POLL)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            self.finished = True

        async def aclose(self) -> None:
            """Part of the real contract; the server does not call it."""
            self.closed = True

        def request_stop(self, reason: str = "") -> None:
            """Record the request. Mirrors the real signature exactly."""
            self.stop_reasons.append(reason)
            self.stopped = True

        def release(self) -> None:
            """Let a held run finish. A no-op once it has already finished."""
            self._release_requested = True

    return Stub, created


def _install_stub(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> tuple[Any, list[Any]]:
    """Build a stub via :func:`make_stub` and patch it in as ``Orchestrator``.

    Returns ``(StubClass, created_instances)``. The *class* is first because the
    server constructs the instance itself, from the symbol it reads at call
    time -- so the only way a test can observe the instance is through the
    instance log, and it must be read *after* the server has built it.
    """
    stub, created = make_stub(**kwargs)
    monkeypatch.setattr(SERVER, "Orchestrator", stub)
    return stub, created


# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _hermetic_web(agent_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ``FURROW_*`` and both API keys, then serve from an empty directory.

    ``agent_env`` guarantees no ambient env var leaks in (and
    ``Settings(_env_file=None)`` would still read the environment, so it is
    required for hermeticity), and ``monkeypatch.chdir`` guarantees a stray
    ``.env`` in the working directory cannot be picked up by the ``Settings()``
    that ``/api/start`` builds without ``_env_file=None``.
    """
    monkeypatch.chdir(agent_env)


@pytest.fixture(autouse=True)
def _no_leaked_runs() -> Iterator[None]:
    """Fail the test that left a run registered -- and clear it regardless.

    ``_active_runs`` is module state that outlives a test. If one scenario
    leaves a run behind, a later "a second concurrent start is rejected" test
    would see a 409 for entirely the wrong reason and still pass. Asserting the
    registry is empty in teardown keeps that from decaying silently; clearing
    it first means even a failing test cannot poison the next one.
    """
    yield
    leaked = dict(SERVER._active_runs)
    SERVER._active_runs.clear()
    assert not leaked, f"run(s) left registered after this test: {sorted(leaked)}"


# ── module identity ────────────────────────────────────────────────────────


def test_web_package_does_not_shadow_the_server_module() -> None:
    """``furrow.web.server`` is reachable as the module, unlike ``furrow.cli.main``.

    ``tests/test_cli.py`` exists because ``furrow/cli/__init__.py`` re-exports
    the click group as ``main``, so ``import furrow.cli.main as m`` binds the
    *Group*. ``furrow/web/__init__.py`` re-exports only ``app`` and ``run``, and
    the name ``server`` is left as the submodule the import system binds, so
    attribute access yields the module. Pinned here because the day someone adds
    a re-export named ``server`` every ``SERVER.Orchestrator`` patch below turns
    into a silent no-op.
    """
    assert isinstance(SERVER, ModuleType)
    assert SERVER is sys.modules["furrow.web.server"]
    assert furrow.web.server is SERVER
    assert SERVER.app is not SERVER
    # The contrast that makes the trap concrete: in furrow.cli the shadowed name
    # really is the re-exported object, not a module.
    assert not isinstance(furrow.cli.main, ModuleType)
    assert isinstance(furrow.cli.main, type(furrow.cli.main))


def test_module_object_patching_replaces_the_orchestrator_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patching ``SERVER.Orchestrator`` is the symbol the server actually reads.

    If this ever stops being true the rest of the file would happily run real
    orchestrators (and, with no API key, real failures) while still passing.
    """
    stub, created = _install_stub(monkeypatch)
    assert SERVER.Orchestrator is stub
    with TestClient(SERVER.app) as client:
        response = client.post("/api/start", json={"goal": "prove the patch is live"})
    assert response.status_code == 200
    assert len(created) == 1
    assert created[0].goal == "prove the patch is live"
    assert created[0].bus is not None
    assert isinstance(created[0].bus, EventBus)
    _wait_for(lambda: not SERVER._active_runs, what="the run to unregister")


# ── the dashboard page ─────────────────────────────────────────────────────


def test_index_serves_a_complete_dashboard() -> None:
    """``GET /`` returns the single self-contained page the browser needs.

    The page has no static-asset directory and no build step, so if any of this
    is missing the UI is silently broken rather than erroring.
    """
    with TestClient(SERVER.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert len(body) > 5_000, "the dashboard is a whole page, not a stub"
    # WebSocket lifecycle wiring.
    assert "location.protocol" in body, "the ws:// vs wss:// choice"
    assert "onopen" in body and "onclose" in body and "onerror" in body
    # The stop control, its handler, and the action it puts on the wire.
    assert 'id="stop-btn"' in body
    assert "requestStop" in body
    assert 'action: "stop"' in body
    # The start control.
    assert 'id="goal"' in body and 'id="start-btn"' in body


def test_dashboard_never_touches_innerhtml() -> None:
    """The served HTML contains ZERO ``innerHTML`` occurrences.

    This is a security property, not a style preference: every string the page
    renders -- the goal, model output, task descriptions, test summaries --
    originates from a language model, and ``innerHTML`` is the one assignment
    that would let any of it execute as script. The page builds nodes with
    ``document.createElement`` and fills them with ``textContent``; if a future
    change introduces ``innerHTML`` anywhere, this fails.
    """
    with TestClient(SERVER.app) as client:
        body = client.get("/").text

    assert body.count("innerHTML") == 0, "model output could execute as script"
    assert body.count("document.write") == 0
    # ... and the safe path is genuinely present, so the check above cannot be
    # satisfied by an empty or stripped page.
    assert "textContent" in body
    assert "document.createElement" in body


# ── health and status ──────────────────────────────────────────────────────


def test_health_reports_ok() -> None:
    """``GET /health`` is the liveness probe."""
    with TestClient(SERVER.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_reflects_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /api/status`` reports what the run registry holds, then empties.

    The endpoint reads ``orchestrator.cycles`` and ``orchestrator.stopped``, so
    this also pins the two attributes the server depends on.
    """
    _stub, created = _install_stub(monkeypatch, hold=True, cycles=STUB_CYCLES)
    with TestClient(SERVER.app) as client:
        assert client.get("/api/status").json() == {"active": 0, "runs": []}

        client.post("/api/start", json={"goal": "report me"})
        _wait_for(lambda: bool(created) and created[0].started, what="the run to start")

        status = client.get("/api/status")
        assert status.status_code == 200
        assert status.json() == {
            "active": 1,
            "runs": [{"goal": "report me", "cycles": STUB_CYCLES, "stopped": False}],
        }

        created[0].release()
        _wait_for(lambda: not SERVER._active_runs, what="the run to unregister")
        assert client.get("/api/status").json() == {"active": 0, "runs": []}
    assert created[0].stopped is False


# ── POST /api/start ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ({"goal": ""}, "empty"),
        ({"goal": "   "}, "whitespace only"),
        ({"goal": "\n\t "}, "other whitespace"),
        ({}, "field missing entirely"),
    ],
    ids=["empty", "whitespace", "newline", "missing"],
)
def test_start_rejects_an_unusable_goal(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
    why: str,
) -> None:
    """A goal that cannot be worked on never reaches an orchestrator.

    pydantic's field validator strips the goal, so a blank one is a 422 before
    any run is registered. ``why`` is only here to make the parametrisation
    readable in a failure report.
    """
    assert why
    _, created = _install_stub(monkeypatch)
    with TestClient(SERVER.app) as client:
        response = client.post("/api/start", json=body)

    assert response.status_code in (400, 422), f"{why} goal was accepted: {response.text}"
    assert created == [], "an orchestrator was built for an unusable goal"
    assert SERVER._active_runs == {}


def test_start_accepts_a_real_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real goal starts a background run and returns immediately."""
    _, created = _install_stub(monkeypatch, cycles=STUB_CYCLES)
    with TestClient(SERVER.app) as client:
        response = client.post("/api/start", json={"goal": "  ship the thing  "})

        assert response.status_code == 200
        assert response.json() == {"started": True, "goal": "ship the thing"}

        assert len(created) == 1
        # Stripped by the validator before it reaches the orchestrator.
        assert created[0].goal == "ship the thing"
        # The server passes its own Settings so the run uses server config.
        assert created[0].settings is not None

        _wait_for(lambda: not SERVER._active_runs, what="the run to finish")
        assert created[0].started is True


def test_second_concurrent_start_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """One run at a time: the second start is a 409, and builds nothing.

    The registry -- not a lock -- is what enforces this, so the assertion that
    matters is that no second orchestrator was constructed.
    """
    _, created = _install_stub(monkeypatch, hold=True)
    with TestClient(SERVER.app) as client:
        first = client.post("/api/start", json={"goal": "first goal"})
        assert first.status_code == 200

        second = client.post("/api/start", json={"goal": "second goal"})
        assert second.status_code == 409
        assert "first goal" in second.json()["detail"]
        assert len(created) == 1, "the rejected start still built an orchestrator"

        created[0].release()
        _wait_for(lambda: not SERVER._active_runs, what="the first run to unregister")

        # Once the run is gone the registry is usable again.
        assert client.post("/api/start", json={"goal": "second goal"}).status_code == 200
        _wait_for(lambda: len(SERVER._active_runs) == 1, what="the second run to register")
        created[1].release()
        _wait_for(lambda: not SERVER._active_runs, what="the second run to unregister")


# ── the websocket: the headline ────────────────────────────────────────────


def test_websocket_streams_every_event_to_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The socket actually carries the run -- this is the regression test.

    ``websocket_endpoint`` used to contain no ``send_json`` call at all: it built
    a queue, subscribed to the bus, and dropped every frame on the floor, so the
    dashboard could only ever show "Starting...". Counting the frames is what
    makes that unrepeatable.

    Expected wire order is
    ``run_started`` (sent directly) -> the stub's events (forwarded by the pump)
    -> the terminal ``run_completed`` (the pump's ``finally``), so
    :data:`EXPECTED_FRAMES` frames must arrive and the last must be the terminal
    frame.
    """
    _stub, created = _install_stub(
        monkeypatch, events=STREAM_EVENTS, cycles=STUB_CYCLES
    )
    with TestClient(SERVER.app) as client, client.websocket_connect("/ws") as session:
        session.send_json({"goal": "demo"})
        frames = [session.receive_json() for _ in range(EXPECTED_FRAMES)]
        print(
            f"[test_web] observed {len(frames)} JSON frames from /ws "
            f"(expected {EXPECTED_FRAMES})"
        )

        # The count itself, and the shape of every frame on the wire.
        assert len(frames) == EXPECTED_FRAMES
        known_types = {member.value for member in EventType}
        types = [frame["type"] for frame in frames]
        assert set(types) <= known_types, f"unknown frame type in {types}"
        for frame in frames:
            assert isinstance(frame["timestamp"], float)
            assert isinstance(frame["cycle"], int)
            assert isinstance(frame["data"], dict)

        # 1. the server's own run_started, sent directly rather than via the bus.
        assert frames[0]["type"] == EventType.RUN_STARTED.value
        assert frames[0]["data"] == {"goal": "demo", "max_cycles": 0}

        # 2. every event the orchestrator emitted, in emission order.
        assert types[1:-1] == [event.type.value for event in STREAM_EVENTS]

        # 3. the terminal frame, built from the orchestrator's own cycle count.
        assert types[-1] == EventType.RUN_COMPLETED.value
        assert frames[-1]["data"] == {"cycles": STUB_CYCLES, "reason": "stream closed"}

        # The dashboard groups the stream by cycle, so a cycle has to survive.
        assert {frame["cycle"] for frame in frames[1:-1]} == {0, 1, 2}
        assert any(frame["cycle"] > 0 for frame in frames)

        # Model-generated text round-trips verbatim as data; the page renders it
        # with textContent (see test_dashboard_never_touches_innerhtml).
        assert any(
            frame["data"].get("result") == HOSTILE_RESULT for frame in frames
        ), "the hostile task result never arrived"

        assert len(created) == 1
        assert created[0].started is True
        assert created[0].cancelled is False

    assert created[0].finished is True


def test_websocket_rejects_malformed_json_with_a_structured_error() -> None:
    """Bad JSON on the opening frame is answered, not raised at the client.

    ``starlette``'s ``WebSocket.receive_json`` propagates
    :class:`json.JSONDecodeError` (unlike some versions, it does not convert it
    into a disconnect), so the server's handler is what turns garbage into a
    ``run.failed`` frame plus a clean close.
    """
    with TestClient(SERVER.app) as client, client.websocket_connect("/ws") as session:
        session.send_text("{ this is not json")

        frame = session.receive_json()
        assert frame["type"] == EventType.RUN_FAILED.value
        assert "malformed JSON" in frame["data"]["error"]
        assert SERVER._active_runs == {}, "a run started despite the bad frame"

        # The server closes after answering, so the next read is a clean
        # disconnect rather than a hang.
        with pytest.raises(WebSocketDisconnect) as excinfo:
            session.receive_json()
        assert excinfo.value.code == 1000


def test_websocket_reports_malformed_json_mid_run_and_stays_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed *subsequent* frame errors without killing the stream.

    Proves the handler ``continue``s instead of propagating: after the error
    frame the socket still accepts a stop action and reaches the orchestrator.
    """
    _stub, created = _install_stub(monkeypatch, hold=True)
    with TestClient(SERVER.app) as client, client.websocket_connect("/ws") as session:
        session.send_json({"goal": "demo"})
        assert session.receive_json()["type"] == EventType.RUN_STARTED.value

        session.send_text("}{ not json either")
        frame = session.receive_json()
        assert frame["type"] == EventType.RUN_FAILED.value
        assert "malformed JSON" in frame["data"]["error"]

        session.send_json({"action": "stop"})
        _wait_for(
            lambda: bool(created) and bool(created[0].stop_reasons),
            what="the socket to stay usable",
        )

    assert created[0].stop_reasons == ["stopped from web"]


def test_websocket_stop_action_reaches_request_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``{"action": "stop"}`` from the browser reaches the running orchestrator.

    This is the dashboard's ``requestStop`` handler's only way to stop a run.
    """
    _stub, created = _install_stub(monkeypatch, hold=True)
    with TestClient(SERVER.app) as client, client.websocket_connect("/ws") as session:
        session.send_json({"goal": "demo"})
        assert session.receive_json()["type"] == EventType.RUN_STARTED.value
        orchestrator = created[0]
        assert orchestrator.stopped is False

        session.send_json({"action": "stop"})
        _wait_for(lambda: bool(orchestrator.stop_reasons), what="request_stop to be called")

        assert orchestrator.stop_reasons == ["stopped from web"]
        assert orchestrator.stopped is True
        assert len(created) == 1

    # Leaving the block closes the socket, which cancels the still-held run.
    _wait_for(lambda: not SERVER._active_runs, what="the run to unregister on disconnect")
    assert orchestrator.cancelled is True


def test_websocket_disconnect_cancels_the_run_and_unsubscribes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the socket mid-run stops the work and detaches the handler.

    A browser tab closed mid-run used to be able to leave an orchestrator
    running with nobody watching it, and to leave a handler on a live bus. Both
    are asserted here: the run sees the cancellation, and the handler count
    returns to exactly what it was -- proved by subscribing an extra handler of
    our own that must survive, so the check cannot pass by dropping everything.
    """
    _stub, created = _install_stub(monkeypatch, hold=True)
    with TestClient(SERVER.app) as client, client.websocket_connect("/ws") as session:
        session.send_json({"goal": "demo"})
        assert session.receive_json()["type"] == EventType.RUN_STARTED.value

        orchestrator = created[0]
        bus = orchestrator.bus
        assert isinstance(bus, EventBus)
        assert bus.handler_count == 1
        # Our own handler, subscribed after the server's, which must be left
        # alone when the server unsubscribes.
        keep = bus.subscribe(lambda event: None)
        assert bus.handler_count == 2

        session.close()

        _wait_for(
            lambda: orchestrator.cancelled,
            what="the background run task to observe cancellation",
        )
        _wait_for(
            lambda: bus.handler_count == 1,
            what="the server's bus handler to be unsubscribed",
        )

    assert orchestrator.stop_reasons == ["client disconnected"]
    assert bus.handler_count == 1
    keep()  # only our own handler was left; removing it empties the bus.
    assert bus.handler_count == 0
    assert SERVER._active_runs == {}


# ── authentication ─────────────────────────────────────────────────────────


def test_without_a_token_no_authentication_is_required() -> None:
    """Unset ``FURROW_WEB_TOKEN`` means an open server, which is the default."""
    assert os.environ.get(TOKEN_ENV) is None, "a token leaked into the environment"
    with TestClient(SERVER.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/api/status").status_code == 200
        # A stray token is simply ignored, not treated as a mismatch.
        assert client.get("/?token=whatever").status_code == 200
        assert client.get("/", headers={"Authorization": "Bearer whatever"}).status_code == 200


def test_a_token_guards_the_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a token set, ``GET /`` needs it -- as a query param or a bearer.

    The query parameter exists because a browser cannot set headers on a plain
    page load; the bearer form exists for scripted clients. Both must work and
    both wrong forms must fail.
    """
    with web_token(SECRET_TOKEN) as reloaded:
        assert reloaded is SERVER, "reload must keep the module object identity"
        assert SERVER._WEB_TOKEN == SECRET_TOKEN
        with TestClient(SERVER.app) as client:
            assert client.get("/").status_code == 401
            assert client.get("/", headers={"Authorization": "Bearer wrong"}).status_code == 401
            assert client.get("/?token=wrong").status_code == 401

            assert client.get(f"/?token={SECRET_TOKEN}").status_code == 200
            ok = client.get("/", headers={"Authorization": f"Bearer {SECRET_TOKEN}"})
            assert ok.status_code == 200
            # The page embeds the token so its own WebSocket handshake can use
            # it -- that is the whole reason the query form is accepted.
            assert f"token={SECRET_TOKEN}" in ok.text

    assert SERVER._WEB_TOKEN == ""
    with TestClient(SERVER.app) as client:
        assert client.get("/").status_code == 200, "the token was not cleared"


def test_a_token_guards_health() -> None:
    """``/health`` is gated by the bearer header only.

    Deliberate asymmetry, pinned here: only ``/`` and ``/ws`` read ``?token=``,
    because only those two are reachable from a browser context that cannot set
    headers. A health probe sends a header, so a query parameter is not accepted
    there.
    """
    with web_token(SECRET_TOKEN), TestClient(SERVER.app) as client:
        assert client.get("/health").status_code == 401
        assert client.get("/health?token=" + SECRET_TOKEN).status_code == 401
        assert (
            client.get(
                "/health", headers={"Authorization": f"Bearer {SECRET_TOKEN}"}
            ).json()
            == {"status": "ok"}
        )


def test_a_token_guards_the_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    """``/ws`` refuses a wrong or missing token, and accepts the right one."""
    with web_token(SECRET_TOKEN):
        _stub, created = _install_stub(monkeypatch, hold=True)
        with TestClient(SERVER.app) as client:
            for url in ("/ws", "/ws?token=wrong", "/ws?token=" + SECRET_TOKEN[:-1] + "x"):
                with (
                    pytest.raises(WebSocketDisconnect) as excinfo,
                    client.websocket_connect(url),
                ):
                    pass
                assert excinfo.value.code == 4401, f"{url} was not refused"

            # Nothing ran for any of the rejected handshakes.
            assert created == []

            with client.websocket_connect(f"/ws?token={SECRET_TOKEN}") as session:
                session.send_json({"goal": "demo"})
                frame = session.receive_json()
                assert frame["type"] == EventType.RUN_STARTED.value
                assert frame["data"]["goal"] == "demo"
                assert len(created) == 1

        _wait_for(
            lambda: not SERVER._active_runs,
            what="the authorised run to unregister on disconnect",
        )


# ── run() ──────────────────────────────────────────────────────────────────


class _UvicornRecorder:
    """Stands in for the ``uvicorn`` module so ``run()`` never binds a socket."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, app: Any, *, host: str, port: int) -> None:
        """Record the bind instead of performing it."""
        self.calls.append({"app": app, "host": host, "port": port})


def test_run_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run()`` binds 127.0.0.1, so an unconfigured server is not exposed.

    Loopback is the only thing standing between a token-less server and the
    network, so this is asserted at both levels: the declared default and the
    value actually handed to uvicorn.
    """
    parameters = inspect.signature(SERVER.run).parameters
    assert parameters["host"].default == "127.0.0.1"
    assert parameters["port"].default == 8000

    recorder = _UvicornRecorder()
    monkeypatch.setattr(SERVER, "uvicorn", recorder)
    SERVER.run()

    assert recorder.calls == [{"app": SERVER.app, "host": "127.0.0.1", "port": 8000}]
    assert recorder.calls[0]["host"] != "0.0.0.0"
    assert json.dumps(recorder.calls[0]["host"])  # plain string, no surprise type
