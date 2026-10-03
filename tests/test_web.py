"""Tests for the HTTP + websocket surface in :mod:`furrow.web.server`.

Every run here is driven by a fake orchestrator with the real constructor
signature: no LLM call, no API key, no network and no real port.  ``TestClient``
runs the app in an in-process portal, so the ``asyncio`` task wrapping
:meth:`RunManager._execute` behaves exactly as it does in production.

Timing is controlled by a :class:`threading.Event` gate the fake waits on between
cycles.  A run therefore stays *in flight* for as long as a test wants it to,
which is what makes the 409 conflict, the stop path and the websocket stream
deterministic instead of a race against a fast fake.  Every test releases the
gate, and the fixture releases it again before the loop shuts down so no run can
outlive the event loop that owns it.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from furrow.config import Settings
from furrow.core.events import Event, EventBus
from furrow.state import RunState, StateStore
from furrow.web import server
from furrow.web.server import RunManager

RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")
TIMEOUT = 5.0
EVENT_SEQUENCE = [
    "run_start",
    "cycle_start",
    "test_result",
    "cycle_start",
    "test_result",
]


# --------------------------------------------------------------------------- fake


@dataclass
class _Harness:
    """Per-test controls shared with the fake orchestrator class."""

    #: Set while the fake is allowed to advance; cleared to hold a run in flight.
    gate: threading.Event = field(default_factory=threading.Event)
    #: When set, the fake raises ``RuntimeError(self.fail)`` straight after run_start.
    fail: str | None = None
    #: Every orchestrator the server built, in creation order.
    created: list[_FakeOrchestrator] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.release()

    def hold(self) -> None:
        """Freeze the fake mid-run until :meth:`release` is called."""
        self.gate.clear()

    def release(self) -> None:
        self.gate.set()


class _FakeOrchestrator:
    """Stand-in for :class:`furrow.core.orchestrator.Orchestrator`.

    Same constructor arguments, same ``goal`` / ``events`` / ``settings`` /
    ``store`` attributes, an awaitable :meth:`run` returning a :class:`RunState`
    and a cooperative :meth:`stop` -- but no model calls and no disk writes.
    """

    _harness: ClassVar[_Harness]

    def __init__(
        self,
        goal: str,
        client: object = None,
        settings: Settings | None = None,
        store: StateStore | None = None,
        events: EventBus | None = None,
        console: object = None,
    ) -> None:
        self.goal = goal
        self.client = client
        self.settings = settings
        self.store = store
        self.events = events if events is not None else EventBus()
        self.console = console
        self.state = RunState(goal=goal)
        self.stop_requested = False
        self._harness.created.append(self)

    async def run(self) -> RunState:
        self.events.emit(Event(kind="run_start", message=f"Goal: {self.goal}"))
        if self._harness.fail is not None:
            raise RuntimeError(self._harness.fail)
        for cycle in (1, 2):
            self.events.emit(Event(kind="cycle_start", cycle=cycle, message=f"Cycle {cycle}"))
            self.events.emit(
                Event(
                    kind="test_result",
                    cycle=cycle,
                    data={"passed": True, "summary": "1 passed"},
                )
            )
            await self._hold()
            self.state.cycles = cycle
        self.state.done = True
        return self.state

    async def _hold(self) -> None:
        """Block until the test lets the run move on, without blocking the loop."""
        gate = self._harness.gate
        if not gate.is_set():
            await asyncio.to_thread(gate.wait)

    async def stop(self) -> None:
        self.stop_requested = True
        self.events.emit(Event(kind="notice", message="stop requested"))
        self._harness.release()


def _bind_harness(harness: _Harness) -> type[_FakeOrchestrator]:
    class _Bound(_FakeOrchestrator):
        _harness = harness

    return _Bound


# ------------------------------------------------------------------------ fixtures


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    """Fresh run manager and a fake orchestrator for one test."""
    control = _Harness()
    monkeypatch.setattr(server, "manager", RunManager())
    monkeypatch.setattr(server, "Orchestrator", _bind_harness(control))
    yield control
    # Backstop: never leave a run parked on the gate once the manager is restored.
    control.release()


@pytest.fixture
def client(harness: _Harness) -> TestClient:
    """A ``TestClient`` whose runs are all released before the loop shuts down."""
    with TestClient(server.app) as test_client:
        try:
            yield test_client
        finally:
            harness.release()


# -------------------------------------------------------------------------- helpers


def _wait_until(predicate: Any, message: str, timeout: float = TIMEOUT) -> None:
    """Poll ``predicate`` from this thread until it is true, or fail loudly."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    if predicate():
        return
    raise AssertionError(message)


def _start(client: TestClient, goal: str, **extra: object) -> dict[str, Any]:
    response = client.post("/api/runs", json={"goal": goal, **extra})
    assert response.status_code == 202, response.text
    body: dict[str, Any] = response.json()
    return body


def _record(run_id: str) -> Any:
    record = server.manager.get(run_id)
    assert record is not None, f"run {run_id} is not in the manager"
    return record


def _await_finished(run_id: str) -> Any:
    record = _record(run_id)
    _wait_until(lambda: record.done, f"run {run_id} never finished")
    return record


def _drain(socket: WebSocketTestSession) -> list[dict[str, Any]]:
    """Read frames until the server closes the socket."""
    frames: list[dict[str, Any]] = []
    while True:
        try:
            frames.append(socket.receive_json())
        except WebSocketDisconnect:
            return frames


def _expect_close(socket: WebSocketTestSession) -> None:
    with pytest.raises(WebSocketDisconnect):
        socket.receive_json()


# ----------------------------------------------------------------------- read only


def test_health_endpoint_reports_ok(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_a_self_contained_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    # Output is appended to this node and nothing is ever assigned as markup.
    assert 'id="out"' in html
    assert "innerHTML" not in html
    assert "textContent" in html
    assert 'document.getElementById("out")' in html
    # A page served over TLS must open a secure socket.
    assert 'location.protocol === "https:"' in html
    assert '"wss:"' in html
    assert '"ws:"' in html
    # The Stop control has to exist for the stop endpoint to be reachable at all.
    assert 'id="stop"' in html
    assert ">Stop<" in html
    assert "/api/runs" in html


# -------------------------------------------------------------------------- start


def test_create_run_returns_202_with_a_run_id_and_a_state(client: TestClient) -> None:
    response = client.post("/api/runs", json={"goal": "build a thing"})
    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"run_id", "state"}
    assert RUN_ID_PATTERN.match(body["run_id"])
    # Before the first save there is no state yet, so only the goal is reported.
    assert body["state"] == {"goal": "build a thing"}


def test_create_run_without_overrides_uses_the_global_settings(
    client: TestClient, harness: _Harness
) -> None:
    _start(client, "no overrides")
    assert harness.created
    assert harness.created[-1].settings is server.settings


def test_create_run_copies_settings_instead_of_mutating_the_singleton(
    client: TestClient, harness: _Harness
) -> None:
    before_model = server.settings.model
    before_cycles = server.settings.max_cycles
    harness.hold()
    try:
        _start(client, "override me", model="  custom-model  ", max_cycles=7)
        assert harness.created
        fake = harness.created[-1]
        assert fake.settings is not None
        # The blank padded model is trimmed by the per-run settings copy.
        assert fake.settings.model == "custom-model"
        assert fake.settings.max_cycles == 7
        assert fake.store is not None
        assert fake.store.path == fake.settings.resolved_state_path
        assert server.settings.model == before_model
        assert server.settings.max_cycles == before_cycles
    finally:
        harness.release()


def test_create_run_rejects_a_missing_goal(client: TestClient) -> None:
    response = client.post("/api/runs", json={"model": "gpt"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "goal"]


def test_create_run_accepts_an_empty_goal_which_only_the_socket_rejects(
    client: TestClient,
) -> None:
    # Recorded on purpose: REST has no minimum length on ``goal`` while the
    # websocket start frame rejects a blank one.
    response = client.post("/api/runs", json={"goal": ""})
    assert response.status_code == 202
    assert response.json()["state"] == {"goal": ""}


# ------------------------------------------------------------------------- slot


def test_second_start_is_rejected_while_a_run_is_active(
    client: TestClient, harness: _Harness
) -> None:
    harness.hold()
    try:
        first = _start(client, "first")
        conflict = client.post("/api/runs", json={"goal": "second"})
        assert conflict.status_code == 409
        assert conflict.json() == {"detail": "a run is already active"}
        assert _record(first["run_id"]).done is False
    finally:
        harness.release()


def test_slot_is_released_once_the_run_finishes(client: TestClient, harness: _Harness) -> None:
    harness.hold()
    try:
        first = _start(client, "first")
        assert _record(first["run_id"]).done is False
    finally:
        harness.release()
    _await_finished(first["run_id"])
    second = _start(client, "second")
    assert second["run_id"] != first["run_id"]


# --------------------------------------------------------------------- reads


def test_get_run_reports_an_unknown_run_as_404(client: TestClient) -> None:
    response = client.get("/api/runs/deadbeefdead")
    assert response.status_code == 404
    assert response.json() == {"detail": "unknown run"}


def test_get_run_returns_the_finished_state(client: TestClient) -> None:
    run_id = _start(client, "inspect me")["run_id"]
    _await_finished(run_id)
    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    state = response.json()
    assert state["goal"] == "inspect me"
    assert state["done"] is True
    assert state["cycles"] == 2


def test_list_runs_is_newest_first_with_a_fixed_key_set(client: TestClient) -> None:
    first = _start(client, "first goal")
    _await_finished(first["run_id"])
    second = _start(client, "second goal")
    _await_finished(second["run_id"])
    response = client.get("/api/runs")
    assert response.status_code == 200
    runs = response.json()["runs"]
    assert [run["run_id"] for run in runs] == [second["run_id"], first["run_id"]]
    assert set(runs[0]) == {"run_id", "goal", "done", "cycles", "halted_reason", "started_at"}
    assert runs[0]["goal"] == "second goal"
    assert runs[0]["done"] is True
    assert runs[0]["cycles"] == 2
    assert runs[0]["halted_reason"] is None
    assert runs[0]["started_at"]


# ------------------------------------------------------------------------ stop


def test_stop_on_a_live_run_reports_stopped_true(client: TestClient, harness: _Harness) -> None:
    harness.hold()
    try:
        run_id = _start(client, "stoppable")["run_id"]
        response = client.post(f"/api/runs/{run_id}/stop")
        assert response.status_code == 200
        assert response.json() == {"stopped": True}
        assert harness.created[-1].stop_requested is True
        _wait_until(lambda: _record(run_id).done, "run never wound down after stop")
    finally:
        harness.release()


def test_stop_on_a_finished_run_reports_stopped_false(client: TestClient) -> None:
    run_id = _start(client, "already done")["run_id"]
    _await_finished(run_id)
    response = client.post(f"/api/runs/{run_id}/stop")
    assert response.status_code == 200
    assert response.json() == {"stopped": False}


def test_stop_on_an_unknown_run_is_404(client: TestClient) -> None:
    response = client.post("/api/runs/deadbeefdead/stop")
    assert response.status_code == 404
    assert response.json() == {"detail": "unknown run"}


# ------------------------------------------------------------------- websocket


def test_websocket_start_frame_streams_events_then_the_final_state(
    client: TestClient, harness: _Harness
) -> None:
    with client.websocket_connect("/ws") as socket:
        socket.send_json({"goal": "stream me"})
        frames = _drain(socket)
    # The regression this endpoint existed to fix: a start frame must produce a
    # stream, not silence.
    assert [frame["kind"] for frame in frames] == [*EVENT_SEQUENCE, "state"]
    assert set(frames[0]) == {"kind", "cycle", "message", "data"}
    assert frames[1]["cycle"] == 1
    assert frames[3]["cycle"] == 2
    assert frames[2]["data"] == {"passed": True, "summary": "1 passed"}
    terminal = frames[-1]
    assert set(terminal) == {"kind", "run_id", "state"}
    assert terminal["kind"] == "state"
    assert RUN_ID_PATTERN.match(terminal["run_id"])
    assert terminal["state"]["goal"] == "stream me"
    assert terminal["state"]["cycles"] == 2
    assert terminal["state"]["done"] is True
    # ``_drain`` only returns once ``receive_json`` raised ``WebSocketDisconnect``,
    # so the terminal frame above was followed by the server closing the socket.


def test_websocket_attaches_to_an_existing_run_with_run_id(
    client: TestClient, harness: _Harness
) -> None:
    harness.hold()
    try:
        run_id = _start(client, "watch this run")["run_id"]
        with client.websocket_connect(f"/ws?run_id={run_id}") as socket:
            harness.release()
            frames = _drain(socket)
    finally:
        harness.release()
    assert [frame["kind"] for frame in frames][-1] == "state"
    assert frames[-1]["run_id"] == run_id
    assert frames[-1]["state"]["done"] is True
    # Events emitted before this socket subscribed are gone, the rest arrive in order.
    received = [frame["kind"] for frame in frames][:-1]
    assert received == EVENT_SEQUENCE[len(EVENT_SEQUENCE) - len(received) :]


def test_websocket_rejects_an_unknown_run_id(client: TestClient) -> None:
    with client.websocket_connect("/ws?run_id=deadbeefdead") as socket:
        assert socket.receive_json() == {"kind": "error", "message": "unknown run"}
        _expect_close(socket)


def test_websocket_requires_a_goal_frame(client: TestClient) -> None:
    for frame in ({}, {"goal": "   "}):
        with client.websocket_connect("/ws") as socket:
            socket.send_json(frame)
            assert socket.receive_json() == {"kind": "error", "message": "a goal is required"}
            _expect_close(socket)


def test_websocket_rejects_a_second_run(client: TestClient, harness: _Harness) -> None:
    harness.hold()
    try:
        _start(client, "holds the slot")
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"goal": "second"})
            assert socket.receive_json() == {
                "kind": "error",
                "message": "a run is already active",
            }
            _expect_close(socket)
    finally:
        harness.release()


def test_websocket_reports_a_failing_orchestrator_and_frees_the_slot(
    client: TestClient, harness: _Harness
) -> None:
    harness.fail = "planner exploded"
    try:
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"goal": "will fail"})
            frames = _drain(socket)
        assert [frame["kind"] for frame in frames] == ["run_start", "error", "state"]
        assert frames[1]["message"] == "RuntimeError: planner exploded"
        # A crashed run has no state to report, only the goal it was given.
        assert frames[2]["state"] == {"goal": "will fail"}
        _wait_until(lambda: server.manager.active is None, "slot stayed taken after a crash")
        # And the slot is genuinely reusable.
        harness.fail = None
        assert _start(client, "next run")["run_id"]
    finally:
        harness.fail = None


def test_websocket_disconnect_unsubscribes_the_client(
    client: TestClient, harness: _Harness
) -> None:
    harness.hold()
    try:
        run_id = _start(client, "vanishing tab")["run_id"]
        record = _record(run_id)
        with client.websocket_connect(f"/ws?run_id={run_id}"):
            _wait_until(lambda: record.subscribers, "the server never subscribed this socket")
            assert len(record.orchestrator.events.subscribers) == 1
        _wait_until(lambda: not record.subscribers, "the server never unsubscribed")
        assert record.orchestrator.events.subscribers == []
    finally:
        harness.release()