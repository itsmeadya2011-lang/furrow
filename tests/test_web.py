"""Regression tests for the ``furrow.web.server`` front end.

The server keeps three process-global registries (``_RUNS``, ``_RUN_TASKS``,
``_STREAMS``) and drives every run from an ``asyncio`` task on the app's event
loop.  The ``TestClient`` runs that loop in its own thread, so a background run
cannot be observed instantaneously: ``_wait_until`` is how the tests assert on
progress, and ``_drain`` is how they read a websocket stream.

Nothing here touches a real model, a real socket or a real key: ``FakeOrchestrator``
stands in for the orchestrator (the endpoints resolve ``server.Orchestrator`` as a
module global at call time, so patching that attribute is enough) and ``TestClient``
runs the ASGI app in-process.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from furrow.config import Settings
from furrow.config import TestResult as _TestResult
from furrow.config import settings as config_settings
from furrow.web import server

#: Exactly the seven keys ``Orchestrator.status()`` promises, plus the ``run_id``
#: ``server._snapshot`` wraps them in.
STATUS_KEYS = {
    "run_id",
    "goal",
    "directive",
    "cycles",
    "running",
    "done",
    "tasks",
    "last_test",
}

#: The event sequence ``FakeOrchestrator.run()`` emits. Asserting the whole list --
#: never a frame count -- is what proves the server forwarded every event it
#: published, in order, dropping and duplicating none.
FAKE_EVENT_SEQUENCE = [
    "cycle_start",
    "plan",
    "task_start",
    "task_end",
    "test_result",
    "cycle_end",
    "run_complete",
]


# --------------------------------------------------------------------------- fakes


class FakeOrchestrator:
    """Stand-in for ``furrow.core.orchestrator.Orchestrator``.

    ``server`` uses exactly four things from the orchestrator: the constructor
    keyword shape, a *synchronous* ``on_event`` callback, the ``cycles`` counter
    (quoted in the ``run_halt`` frame) and ``status()``.  This fake honours that
    contract exactly and involves no LLM, no filesystem and no clock beyond
    ``asyncio.sleep``, which keeps the tests deterministic and offline.
    """

    #: Gap between two consecutive events.  Slowed down per test (via
    #: ``monkeypatch.setattr``) when a run has to outlive the test's own setup
    #: handshake, e.g. when a second socket attaches after the run has started.
    step_delay = 0.005

    def __init__(
        self,
        goal: str,
        client: Any = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        settings: Any = None,
    ) -> None:
        self.goal = goal
        self.directive = goal
        self.client = client
        self.on_event = on_event
        self.settings = settings
        self.cycles = 0
        self.running = False
        self.done = False
        self.completed = False
        self.cancelled = False
        self.tasks: list[dict[str, Any]] = []
        self.last_test: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []

    def _emit(self, event_type: str, message: str, data: dict[str, Any] | None = None) -> None:
        event = {
            "type": event_type,
            "cycle": self.cycles,
            "message": message,
            "data": dict(data or {}),
        }
        self.events.append(event)
        if self.on_event is not None:
            # Sync, exactly like the real orchestrator's ``_emit``.
            self.on_event(event)

    def status(self) -> dict[str, Any]:
        """The seven keys the REST API and the bundled page both depend on."""
        return {
            "goal": self.goal,
            "directive": self.directive,
            "cycles": self.cycles,
            "running": self.running,
            "done": self.done,
            "tasks": list(self.tasks),
            "last_test": self.last_test,
        }

    async def _tick(self) -> None:
        await asyncio.sleep(self.step_delay)

    async def run(self) -> None:
        self.running = True
        try:
            # The sleep before the *first* emit is load-bearing, not decoration.
            # BUG WORKAROUND: ``_start_run`` schedules this coroutine
            # (server.py:176) before the /ws handler reaches
            # ``_RunStream.subscribe()`` (server.py:252), and ``publish()`` only
            # fans out to the subscribers that exist at that instant -- so an
            # event emitted before the handler subscribes is lost for good. A
            # zero-delay fake is exactly how that race shows up.
            await self._tick()
            self.cycles += 1
            self._emit(
                "cycle_start",
                f"Cycle {self.cycles} started",
                {"goal": self.goal, "directive": self.directive},
            )
            await self._tick()
            self._emit(
                "plan",
                "Planned 1 task(s)",
                {"rationale": "one task is enough", "tasks": [_TASK]},
            )
            for event_type, message, data in (
                ("task_start", "Task t1 started", {"id": "t1", "description": _DESCRIPTION}),
                (
                    "task_end",
                    "Task t1 completed",
                    {"id": "t1", "status": "completed", "result": "done"},
                ),
            ):
                await self._tick()
                self._emit(event_type, message, data)
            result = _TestResult(passed=True, summary="1 passed, 0 failed", failures=[])
            self.last_test = {
                "passed": result.passed,
                "summary": result.summary,
                "failures": list(result.failures),
            }
            await self._tick()
            self._emit("test_result", result.summary, dict(self.last_test))
            self.tasks = [{**_TASK, "status": "completed", "result": "done"}]
            await self._tick()
            self._emit(
                "cycle_end",
                "Cycle 1 complete",
                {"passed": True, "completed": 1, "failed": 0},
            )
            self.done = True
            await self._tick()
            self._emit(
                "run_complete",
                "Goal complete.",
                {"cycles": self.cycles, "summary": result.summary},
            )
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.running = False
            # Only ever True once the whole body has run to the end, so a test can
            # poll it and then assert on every other flag without a race.
            self.completed = not self.cancelled and self.done


_DESCRIPTION = "ship the thing"
_TASK = {"id": "t1", "description": _DESCRIPTION}


# -------------------------------------------------------------------------- helpers


def _wait_until(predicate: Callable[[], Any], timeout: float = 5.0) -> None:
    """Block until ``predicate()`` is truthy, or fail loudly naming the timeout.

    Runs progress on the TestClient's own event loop thread, so this spins with
    ``time.sleep`` (which releases the GIL) until the background work lands.
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        time.sleep(0.01)


def _drain(ws: WebSocketTestSession) -> list[dict[str, Any]]:
    """Every frame the server sends on ``ws`` until it closes the connection.

    BUG LOCK -- read until the close, never a fixed frame count.  The number of
    frames is not part of the wire contract: it depends on how many events the
    loop produced, which depends on the model.  An earlier smoke test read N
    frames and reported a failure whenever the run produced a different N, which
    is a false failure in the test, never a defect in the server.
    """
    frames: list[dict[str, Any]] = []
    while True:
        try:
            frames.append(ws.receive_json())
        except WebSocketDisconnect:
            return frames


def _event_types(frames: list[dict[str, Any]]) -> list[str]:
    return [frame["type"] for frame in frames]


async def _cancel_leftover_runs() -> None:
    """Cancel every run task a test left in flight and let the loop observe it.

    Called through ``client.portal.call`` because the app's event loop lives in
    the TestClient's own thread; cancelling a task from this thread is a race.
    """
    tasks = [task for task in server._RUN_TASKS.values() if not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ------------------------------------------------------------------------- fixtures


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # The endpoints look ``Orchestrator`` up as a module global at call time, so
    # patching the module attribute is all that is needed -- no dependency
    # overrides, and no real LLM client is ever constructed.
    monkeypatch.setattr(server, "Orchestrator", FakeOrchestrator)
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def isolated_run_registry(client: TestClient) -> Iterator[None]:
    """Keep the three server registries from leaking between tests.

    They are process-global, so without this an earlier test's run would show up
    in ``GET /api/runs`` for a later one, and its background task would be
    destroyed mid-flight when the event loop closed ("Task was destroyed but it
    is pending!").  Cleared on both sides: before, because another test module may
    have left entries behind; after, so this module leaks nothing.
    """
    server._RUNS.clear()
    server._RUN_TASKS.clear()
    server._STREAMS.clear()
    yield
    client.portal.call(_cancel_leftover_runs)
    server._RUNS.clear()
    server._RUN_TASKS.clear()
    server._STREAMS.clear()


# ------------------------------------------------------------------------ rest api


def test_post_api_runs_returns_202_with_a_run_id_and_the_status_keys(client):
    response = client.post("/api/runs", json={"goal": "  add a /health endpoint  "})

    assert response.status_code == 202
    body = response.json()
    assert set(body) == STATUS_KEYS
    assert body["run_id"] in server._RUNS
    # The goal is stripped before the run is created.
    assert body["goal"] == "add a /health endpoint"
    assert body["directive"] == "add a /health endpoint"
    assert body["cycles"] == 0
    assert body["tasks"] == []
    assert body["last_test"] is None


def test_get_api_runs_lists_the_run_it_created(client):
    # BUG LOCK: the old /ws drove a run inline and registered nothing, so the
    # run list was permanently empty and there was no way to rejoin a run.
    run_id = client.post("/api/runs", json={"goal": "list me"}).json()["run_id"]

    response = client.get("/api/runs")

    assert response.status_code == 200
    entries = response.json()
    assert {entry["run_id"] for entry in entries} == {run_id}
    assert entries[0]["goal"] == "list me"
    assert set(entries[0]) == STATUS_KEYS


def test_get_api_runs_returns_the_status_of_a_single_run(client):
    created = client.post("/api/runs", json={"goal": "inspect me"}).json()

    response = client.get(f"/api/runs/{created['run_id']}")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == STATUS_KEYS
    assert body["run_id"] == created["run_id"]
    assert body["goal"] == "inspect me"


def test_get_api_runs_rejects_an_unknown_run_id_with_404(client):
    response = client.get("/api/runs/does-not-exist")

    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_model_override_applies_to_that_run_only_and_leaves_the_singleton_alone(client):
    # BUG LOCK -- global configuration mutated from a request.  ``StartRequest``
    # carries an optional ``model``; the server applies it by copying the
    # process-wide ``furrow.config.settings`` singleton
    # (``default_settings.model_copy(update={"model": model})`` in
    # ``_settings_for``) and handing the copy to that one orchestrator as
    # ``settings=``.  Writing the override back into the singleton instead would
    # let one browser tab re-point every other concurrent run -- and the CLI --
    # at its model, permanently, with no way to undo it.
    before = config_settings.model
    assert before != "other-model", "the override has to differ from the default"

    created = client.post("/api/runs", json={"goal": "use another model", "model": "other-model"})
    assert created.status_code == 202
    run = server._RUNS[created.json()["run_id"]]

    # The override reached this run...
    assert run.settings.model == "other-model"
    # ...on its own Settings copy, not on the shared singleton...
    assert isinstance(run.settings, Settings)
    assert run.settings is not config_settings
    # ...built with model_copy, so every other field still comes from the
    # singleton rather than being reset to a field default.
    assert run.settings.max_cycles == config_settings.max_cycles
    # ...and the singleton is byte-for-byte unchanged.
    assert config_settings.model == before

    # "That run only": a second run with no override still gets the global model.
    other = server._RUNS[client.post("/api/runs", json={"goal": "no override"}).json()["run_id"]]
    assert other.settings is None
    assert config_settings.model == before


def test_post_api_runs_rejects_an_empty_goal_instead_of_starting_a_doomed_run(client):
    # BUG LOCK: a blank goal used to create a run that could never succeed, and
    # then sat there burning the cycle budget with nothing to plan against.
    response = client.post("/api/runs", json={"goal": "   \n\t "})

    assert response.status_code == 422
    assert "goal" in response.json()["detail"]
    assert server._RUNS == {}


# ------------------------------------------------------------------------ websockets


def test_websocket_streams_a_control_frame_then_every_event_until_run_complete(client):
    # BUG LOCK -- the silent /ws.  The first /ws handler awaited the unbounded
    # ``Orchestrator.run()`` inline, so the connection was accepted and then not a
    # single frame was ever written to it: the bundled console rendered an empty
    # log for the whole run and the operator had no way to see any progress.  The
    # fix streams the orchestrator's events from a background task; this pins the
    # wire contract that makes the page work again.
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "add a /health endpoint"})
        frames = _drain(ws)

    control, *events = frames
    assert control["type"] == "run_started"
    assert control["run_id"] in server._RUNS
    assert control["status"]["goal"] == "add a /health endpoint"

    # Every published event arrives, in order, and the last one is terminal.
    assert _event_types(events) == FAKE_EVENT_SEQUENCE
    assert all({"type", "cycle", "message", "data"} <= set(event) for event in events)
    assert "run_halt" not in _event_types(events)


def test_run_keeps_going_after_the_client_disconnects_mid_stream(monkeypatch, client):
    # BUG LOCK: detaching used to be indistinguishable from cancelling.  A run
    # lives in its own task, so closing the socket may only unsubscribe the
    # socket -- the run has to finish on its own.
    monkeypatch.setattr(FakeOrchestrator, "step_delay", 0.04)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "keep going after I leave"})
        started = ws.receive_json()
    # Leaving the `with` closes the socket while the run is still going.
    run = server._RUNS[started["run_id"]]

    _wait_until(lambda: run.completed)

    assert run.cancelled is False
    assert run.running is False
    assert run.done is True
    assert _event_types(run.events) == FAKE_EVENT_SEQUENCE
    assert server._STREAMS[started["run_id"]] is not None


def test_attaching_by_run_id_streams_that_run(monkeypatch, client):
    # The page has an Attach button so a second observer -- or a reloaded tab --
    # can watch a run it did not start.  Both sockets see the same stream.
    monkeypatch.setattr(FakeOrchestrator, "step_delay", 0.04)
    run_id = client.post("/api/runs", json={"goal": "watch me"}).json()["run_id"]

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"run_id": run_id})
        frames = _drain(ws)

    control, *events = frames
    assert control["type"] == "run_attached"
    assert control["run_id"] == run_id
    assert control["status"]["goal"] == "watch me"
    assert _event_types(events) == FAKE_EVENT_SEQUENCE


def test_attaching_to_an_unknown_run_id_answers_with_an_error_frame(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"run_id": "does-not-exist"})
        frame = ws.receive_json()
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()

    assert frame["type"] == "error"
    assert "does-not-exist" in frame["error"]
    # The message is mirrored on both keys so the page can read either one.
    assert frame["error"] == frame["message"]
    assert server._RUNS == {}


def test_websocket_rejects_a_first_frame_with_neither_goal_nor_run_id(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "   "})
        frame = ws.receive_json()
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()

    assert frame["type"] == "error"
    # A blank goal must not create a run that can never succeed.
    assert server._RUNS == {}


def test_two_sockets_can_attach_to_one_run_and_both_see_every_event(monkeypatch, client):
    monkeypatch.setattr(FakeOrchestrator, "step_delay", 0.04)
    with client.websocket_connect("/ws") as first:
        first.send_json({"goal": "two watchers"})
        started = first.receive_json()
        with client.websocket_connect("/ws") as second:
            second.send_json({"run_id": started["run_id"]})
            attached = second.receive_json()
            # Both control frames were read above, so these two drains start at the
            # first orchestrator event.
            first_frames = _drain(first)
            second_frames = _drain(second)

    assert started["type"] == "run_started"
    assert attached["type"] == "run_attached"
    assert attached["run_id"] == started["run_id"]
    assert _event_types(first_frames) == FAKE_EVENT_SEQUENCE
    assert _event_types(second_frames) == FAKE_EVENT_SEQUENCE
    # Same events, byte for byte, from both subscribers of the one fan-out.
    assert first_frames == second_frames


# ---------------------------------------------------------------------------- page


def test_index_page_has_the_goal_input_and_the_log(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert 'id="goal"' in html
    assert 'id="log"' in html
    assert 'id="start"' in html
    assert 'id="attach"' in html


def test_index_page_never_parses_server_or_model_text_as_html(client):
    # BUG LOCK -- the XSS-shaped hole in the first version of the page: it built
    # the log with ``logEl.innerHTML += message``, so anything a model emitted (a
    # ``<script>`` block, an ``<img onerror=...>``) executed in the operator's own
    # browser.  Every server- and model-derived string now goes through a text
    # sink, so none of the HTML sinks may come back.
    html = client.get("/").text

    assert "innerHTML" not in html
    assert "outerHTML" not in html
    assert "insertAdjacentHTML" not in html
    assert "document.write" not in html
    assert "textContent" in html


def test_index_page_loads_no_external_assets(client):
    # Everything is inlined on purpose: a CDN reference would ship the operator's
    # goals and the model's output to a third party, and would leave the console
    # blank whenever the machine is offline.  Checked as ``"http://"`` /
    # ``"https://"`` rather than ``"https:"`` -- the page legitimately contains
    # ``'https:'`` as a JS comparison operand and ``ws://``/``wss://`` in its
    # socket-URL builder, neither of which is an asset reference.
    html = client.get("/").text

    assert "http://" not in html
    assert "https://" not in html
    assert "<script src" not in html
    assert "<link" not in html
    assert "@import" not in html
