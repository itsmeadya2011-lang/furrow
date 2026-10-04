"""Tests for the HTTP + WebSocket surface in :mod:`furrow.web.server`.

HEADLINE REGRESSION (bug #11): the original ``/ws`` handler accepted the goal, awaited
``orchestrator.run()`` and never called ``websocket.send_json``. The bundled UI's
``ws.onmessage`` therefore fired zero times and live monitoring showed only
``"Starting..."`` followed by ``"Closed."`` -- a whole feature was dead. Every websocket
test below therefore *reads frames back off the socket* and asserts that several real event
frames arrive between a ``started`` and a ``done`` control frame. Zero frames is exactly
the bug, so ``assert len(event_frames) > 1`` is the load-bearing line.

Nothing here binds a port or touches the network: ``TestClient`` drives the ASGI app
in-process on its own event loop, and :class:`FakeOrchestrator` replaces the real
orchestrator, so no LLM, workspace or shell command is involved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from furrow.config import Settings, get_settings
from furrow.core.orchestrator import RunSummary
from furrow.events import EventBus, EventType
from furrow.web import server as server_module
from furrow.web.server import MAX_ACTIVE_RUNS, app, registry

#: Every ``Event.to_dict()`` frame carries exactly these keys.
EVENT_FRAME_KEYS = {"type", "message", "data", "cycle", "timestamp"}

#: Upper bound on frames read while waiting for a terminal ``done`` control frame.
MAX_FRAMES = 64

#: Upper bound on wall-clock waits. Every gate is opened explicitly, so these are only
#: ever reached when the code under test stops making progress.
GATE_TIMEOUT = 5.0
POLL_INTERVAL = 0.005

TOKEN = "s3cret-token"


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


class _Gate:
    """Cross-thread latch.

    The fake orchestrator awaits it on the app's event loop while a test opens it from the
    pytest thread. That makes "read the frames that are already buffered, then let the rest
    stream live" deterministic instead of timing-dependent, without any real sleep.
    """

    def __init__(self, timeout: float = GATE_TIMEOUT) -> None:
        self._flag = threading.Event()
        self._timeout = timeout
        self.timed_out = False

    def open(self) -> None:
        """Release every coroutine waiting on this gate."""
        self._flag.set()

    async def wait(self) -> bool:
        """Return True once opened, False if the budget ran out (never raises)."""
        deadline = time.monotonic() + self._timeout
        while not self._flag.is_set():
            if time.monotonic() >= deadline:
                self.timed_out = True
                return False
            await asyncio.sleep(0.002)
        return True


class _UsageOnlyClient:
    """Stands in for ``LLMClient``: the server reads ``.usage`` off it for snapshots."""

    def __init__(self) -> None:
        self._usage = {"input_tokens": 7, "output_tokens": 3, "requests": 2}

    @property
    def usage(self) -> dict[str, int]:
        return dict(self._usage)


class FakeOrchestrator:
    """LLM-free stand-in for :class:`furrow.core.orchestrator.Orchestrator`.

    Exposes exactly the surface the server touches (``.bus``, ``.cycles``,
    ``.halted_reason``, ``.snapshot()``, ``await .run()``, ``await .stop()``, ``.client.usage``)
    and emits a fixed, recognisable event sequence that deliberately includes one
    ``EventType.ERROR`` -- its ``"error"`` string value collides with the ``type`` of the
    control error frame, which is exactly the ambiguity clients must resolve via ``kind``.

    ``PRE_GATE_EVENTS`` are emitted before the optional ``gate`` is awaited and
    ``POST_GATE_EVENTS`` after it, so a test can read the replayed frames, release the gate
    and then observe genuinely live traffic.
    """

    PRE_GATE_EVENTS = (
        EventType.RUN_START,
        EventType.CYCLE_START,
        EventType.PLAN,
    )
    POST_GATE_EVENTS = (
        EventType.TASK_START,
        EventType.TOOL_CALL,
        EventType.TASK_END,
        EventType.TEST_RESULT,
        EventType.CYCLE_END,
        EventType.ERROR,
        EventType.RUN_END,
    )

    def __init__(
        self,
        goal: str,
        client: Any = None,
        settings: Settings | None = None,
        bus: EventBus | None = None,
        *,
        gate: _Gate | None = None,
        summary: RunSummary | None = None,
        error: BaseException | None = None,
        sleep_before_crash: float = 0.0,
    ) -> None:
        self.goal = goal
        # Mirrors the real orchestrator, which falls back to the process singleton.
        self.settings: Settings = settings if settings is not None else get_settings()
        self.bus = bus if bus is not None else EventBus(500)
        self.client = client if client is not None else _UsageOnlyClient()
        self.gate = gate
        self.cycles = 0
        self.completed = False
        self.stop_calls = 0
        self.error = error
        self.sleep_before_crash = sleep_before_crash
        self.summary = summary if summary is not None else RunSummary(
            cycles=1,
            tasks_total=1,
            tasks_completed=1,
            tasks_failed=0,
            tests_passed=True,
            halted_reason="goal_complete",
        )
        self._halted_reason: str | None = None

    # -- surface the server reads ------------------------------------------
    @property
    def halted_reason(self) -> str | None:
        return self._halted_reason

    def snapshot(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "focus": self.goal,
            "cycles": self.cycles,
            "halted_reason": self._halted_reason,
            "running": not self.completed,
            "tasks": [
                {
                    "id": "1",
                    "description": "Add a greeting module",
                    "status": "completed",
                    "attempts": 1,
                    "files": ["greeting.py"],
                    "dependencies": [],
                    "result": "done",
                }
            ],
            "test_results": [
                {"passed": True, "summary": "scripted", "failures": [], "commands": []}
            ],
            "usage": self.client.usage,
        }

    # -- behaviour ----------------------------------------------------------
    async def run(self) -> RunSummary:
        self.cycles = 1
        try:
            self._emit_all(FakeOrchestrator.PRE_GATE_EVENTS)
            if self.gate is not None:
                await self.gate.wait()
            self._emit_all(FakeOrchestrator.POST_GATE_EVENTS)
            if self.error is not None:
                if self.sleep_before_crash:
                    await asyncio.sleep(self.sleep_before_crash)
                raise self.error
            self._halted_reason = self.summary.halted_reason
            return self.summary
        finally:
            self.completed = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self._halted_reason = "stopped"
        if self.gate is not None:
            # A real stop releases a real run; without this a stoppable test would hang.
            self.gate.open()

    def _emit_all(self, event_types: tuple[EventType, ...]) -> None:
        for index, event_type in enumerate(event_types):
            self.bus.emit(
                event_type,
                f"{event_type.value} message",
                cycle=self.cycles,
                position=index,
            )


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gate: _Gate | None = None,
    summary: RunSummary | None = None,
    error: BaseException | None = None,
    sleep_before_crash: float = 0.0,
) -> list[FakeOrchestrator]:
    """Swap ``furrow.web.server.Orchestrator`` for :class:`FakeOrchestrator`.

    Returns the list the double appends itself to, so a test can count constructions and
    inspect what it was given.
    """
    created: list[FakeOrchestrator] = []

    class Installed(FakeOrchestrator):
        """Per-test subclass so each install carries its own gate and outcome."""

        def __init__(
            self,
            goal: str,
            client: Any = None,
            settings: Settings | None = None,
            bus: EventBus | None = None,
        ) -> None:
            super().__init__(
                goal,
                client=client,
                settings=settings,
                bus=bus,
                gate=gate,
                summary=summary,
                error=error,
                sleep_before_crash=sleep_before_crash,
            )
            created.append(self)

    monkeypatch.setattr(server_module, "Orchestrator", Installed)
    return created


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """The run registry is a module-level singleton; never leak runs between tests."""
    registry.clear()
    yield
    registry.clear()


def wait_until(predicate: Callable[[], bool], timeout: float = GATE_TIMEOUT) -> bool:
    """Poll `predicate` from the pytest thread until it is true or the budget runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(POLL_INTERVAL)
    return predicate()


def receive_frame(websocket: Any, *, timeout: float = GATE_TIMEOUT) -> dict[str, Any]:
    """Read exactly one JSON frame, with a wall-clock budget.

    ``TestClient.receive_json`` blocks the calling thread on an in-process portal, so a
    handler that forwards nothing at all - the exact shape of regression #11 - would hang
    the suite forever instead of failing. Each read therefore runs on a short-lived worker
    thread; overrunning the budget is an explicit, fast failure.
    """
    box: list[dict[str, Any]] = []
    failure: list[BaseException] = []

    def read() -> None:
        try:
            box.append(websocket.receive_json())
        except BaseException as exc:  # re-raised verbatim on the calling thread
            failure.append(exc)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(timeout)
    if reader.is_alive():
        raise AssertionError(
            f"no frame arrived within {timeout}s: the server forwarded nothing to the socket"
        )
    if failure:
        raise failure[0]
    return box[0]


def read_frames(websocket: Any, count: int) -> list[dict[str, Any]]:
    """Read exactly `count` JSON frames off the socket."""
    return [receive_frame(websocket) for _ in range(count)]


def read_until_done(websocket: Any) -> list[dict[str, Any]]:
    """Read frames until the terminal ``done`` control frame arrives."""
    frames: list[dict[str, Any]] = []
    for _ in range(MAX_FRAMES):
        frame = receive_frame(websocket)
        frames.append(frame)
        if frame.get("kind") == "control" and frame.get("type") == "done":
            return frames
    raise AssertionError(f"no done control frame within {MAX_FRAMES} frames: {frames}")


def is_control(frame: dict[str, Any]) -> bool:
    return frame.get("kind") == "control"


def is_event(frame: dict[str, Any]) -> bool:
    return not is_control(frame) and EVENT_FRAME_KEYS <= set(frame)


def event_types(frames: list[dict[str, Any]]) -> list[str]:
    return [frame["type"] for frame in frames if is_event(frame)]


def assert_event_frame(frame: dict[str, Any]) -> None:
    """Every event frame must be exactly ``Event.to_dict()`` and JSON-parseable."""
    assert set(frame) == EVENT_FRAME_KEYS, frame
    assert isinstance(frame["type"], str), frame
    assert isinstance(frame["message"], str), frame
    assert isinstance(frame["data"], dict), frame
    assert isinstance(frame["cycle"], int), frame
    assert isinstance(frame["timestamp"], float), frame
    assert json.loads(json.dumps(frame)) == frame, frame


def start_over_websocket(
    client: TestClient,
    created: list[FakeOrchestrator],
    gate: _Gate,
    *,
    path: str = "/ws",
    goal: str = "demo",
    pre_gate: int = len(FakeOrchestrator.PRE_GATE_EVENTS),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the full START-mode handshake and return ``(before_gate, whole_stream)``.

    Opens `gate` only after the pre-gate frames have been read, which guarantees those
    frames are the replayed history and everything after the gate is live traffic.
    """
    replayed: list[dict[str, Any]] = []
    stream: list[dict[str, Any]] = []
    with client.websocket_connect(path) as websocket:
        websocket.send_json({"goal": goal})
        started = receive_frame(websocket)
        stream.append(started)
        replayed = read_frames(websocket, pre_gate)
        stream.extend(replayed)
        gate.open()
        rest = read_until_done(websocket)
        stream.extend(rest)
    assert created, "the fake orchestrator was never constructed"
    return replayed, stream


# ---------------------------------------------------------------------------
# websocket streaming (bug #11)
# ---------------------------------------------------------------------------


def test_start_mode_streams_event_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """REGRESSION: the socket must actually receive frames.

    The old handler ran the goal and sent nothing, so the UI stayed on "Starting...". Here
    the client reads real frames back: a ``started`` control frame, several verbatim
    ``Event.to_dict()`` event frames, and a ``done`` control frame carrying the summary.
    """
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            replayed, stream = start_over_websocket(client, created, gate)
    finally:
        gate.open()

    event_frames = [frame for frame in stream if is_event(frame)]
    print(f"\nreceived {len(stream)} frames: {[frame.get('type') for frame in stream]}")

    # The regression itself: zero frames is what the bug produced.
    assert len(event_frames) > 1, f"only {len(event_frames)} event frames arrived: {stream}"

    assert is_control(stream[0]), stream[0]
    assert stream[0]["type"] == "started", stream[0]
    assert stream[0]["run_id"], stream[0]
    assert stream[0]["goal"] == "demo", stream[0]

    assert is_control(stream[-1]), stream[-1]
    assert stream[-1]["type"] == "done", stream[-1]
    assert stream[-1]["run_id"] == stream[0]["run_id"], stream[-1]
    assert stream[-1]["summary"] == {
        "cycles": 1,
        "tasks_total": 1,
        "tasks_completed": 1,
        "tasks_failed": 0,
        "tests_passed": True,
        "halted_reason": "goal_complete",
    }, stream[-1]
    assert stream[-1]["error"] is None, stream[-1]

    for frame in event_frames:
        assert_event_frame(frame)

    # Every scripted event type reached the client, in order.
    assert event_types(stream) == [
        *(event_type.value for event_type in FakeOrchestrator.PRE_GATE_EVENTS),
        *(event_type.value for event_type in FakeOrchestrator.POST_GATE_EVENTS),
    ], event_types(stream)

    # The pre-gate frames were replayed history; the post-gate frames are live traffic.
    assert len(replayed) == len(FakeOrchestrator.PRE_GATE_EVENTS), replayed
    assert len(stream) == len(replayed) + len(FakeOrchestrator.POST_GATE_EVENTS) + 2


def test_start_mode_frames_carry_cycles_and_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Event frames must be usable by the UI: real ``cycle`` numbers and ``data`` payloads."""
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            _, stream = start_over_websocket(client, created, gate)
    finally:
        gate.open()

    event_frames = [frame for frame in stream if is_event(frame)]
    assert len(event_frames) > 1
    assert {frame["cycle"] for frame in event_frames} == {1}
    assert all(frame["message"] for frame in event_frames)
    positions = {frame["type"]: frame["data"]["position"] for frame in event_frames}
    assert positions["run_start"] == 0
    assert positions["run_end"] == len(FakeOrchestrator.POST_GATE_EVENTS) - 1
    timestamps = [frame["timestamp"] for frame in event_frames]
    assert timestamps == sorted(timestamps), timestamps


def test_attach_mode_replays_buffered_then_streams_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATTACH mode replays history, then follows live events, and starts nothing."""
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            posted = client.post("/api/runs", json={"goal": "attach me"})
            assert posted.status_code == 202, posted.text
            run_id = posted.json()["run_id"]

            buffered = client.get(f"/api/runs/{run_id}/events").json()["events"]
            assert event_types([dict(event) for event in buffered]) == [
                event_type.value for event_type in FakeOrchestrator.PRE_GATE_EVENTS
            ]
            assert len(created) == 1

            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"run_id": run_id})
                started = receive_frame(websocket)
                assert started["kind"] == "control"
                assert started["type"] == "started"
                assert started["run_id"] == run_id
                replayed = read_frames(websocket, len(FakeOrchestrator.PRE_GATE_EVENTS))
                assert event_types(replayed) == [
                    event_type.value for event_type in FakeOrchestrator.PRE_GATE_EVENTS
                ]
                gate.open()
                live = read_until_done(websocket)
    finally:
        gate.open()

    # Attaching must never construct another orchestrator.
    assert len(created) == 1, [fake.goal for fake in created]

    done = live[-1]
    assert done["kind"] == "control" and done["type"] == "done"
    assert done["run_id"] == run_id
    assert done["summary"]["halted_reason"] == "goal_complete"
    assert event_types(live) == [
        event_type.value for event_type in FakeOrchestrator.POST_GATE_EVENTS
    ], event_types(live)


def test_attach_mode_rejects_unknown_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown ``run_id`` is answered with a control error frame and close code 1008."""
    install_fake(monkeypatch)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"run_id": "does-not-exist"})
            frame = receive_frame(websocket)
            assert frame == {
                "kind": "control",
                "type": "error",
                "error": "unknown run id: does-not-exist",
            }
            # The refusal is immediately followed by the policy close.
            with pytest.raises(WebSocketDisconnect) as closed:
                receive_frame(websocket)
            assert closed.value.code == 1008
        assert registry.runs() == []


def test_websocket_rejects_a_blank_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only goal is refused before anything is constructed (close code 1003)."""
    created = install_fake(monkeypatch)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"goal": "   "})
            frame = receive_frame(websocket)
            assert frame["kind"] == "control"
            assert frame["type"] == "error"
            assert "goal" in frame["error"], frame
            with pytest.raises(WebSocketDisconnect) as closed:
                receive_frame(websocket)
            assert closed.value.code == 1003
    assert created == []
    assert registry.runs() == []


def test_control_frames_are_distinguishable_from_error_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EventType.ERROR.value == "error"`` collides with the control ``type``.

    Both frames therefore carry ``"error"`` as their type; only ``kind`` separates them, and
    only the event frame carries ``data``/``cycle``/``timestamp``.
    """
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            _, stream = start_over_websocket(client, created, gate)
    finally:
        gate.open()

    error_events = [
        frame
        for frame in stream
        if not is_control(frame) and frame.get("type") == EventType.ERROR.value
    ]
    assert len(error_events) == 1, stream
    event_frame = error_events[0]
    assert "kind" not in event_frame, event_frame
    assert_event_frame(event_frame)

    controls = [frame for frame in stream if is_control(frame)]
    assert {frame["type"] for frame in controls} == {"started", "done"}
    for frame in controls:
        assert "data" not in frame, frame
        assert "timestamp" not in frame, frame

    # The dispatch rule the UI relies on: shape, never `type` alone.
    for frame in stream:
        assert is_control(frame) or is_event(frame), frame
        if not is_control(frame):
            assert frame.get("kind") != "control", frame


def test_run_crash_is_reported_in_the_done_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crashing orchestrator must not hang the socket: it yields ERROR frames then ``done``."""
    install_fake(
        monkeypatch,
        error=RuntimeError("boom"),
        sleep_before_crash=0.01,
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"goal": "explode"})
            started = receive_frame(websocket)
            assert started["type"] == "started"
            frames = read_until_done(websocket)

    done = frames[-1]
    assert done["kind"] == "control" and done["type"] == "done"
    assert done["summary"] is None, done
    assert done["error"] is not None and "RuntimeError: boom" in done["error"], done
    error_frames = [
        frame for frame in frames if not is_control(frame) and frame["type"] == "error"
    ]
    assert len(error_frames) == 2, [frame["type"] for frame in frames]


# ---------------------------------------------------------------------------
# REST surface
# ---------------------------------------------------------------------------


def test_health_reports_status_and_version() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["runs_tracked"] == 0
    assert body["runs_running"] == 0
    assert body["auth_required"] is False


def test_index_serves_the_monitoring_ui() -> None:
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "<title>Furrow</title>" in html
    # Stable behavioural substrings only - no markup assertions.
    assert "live agent monitoring" in html
    assert "/ws" in html


def test_index_builds_a_secure_websocket_url() -> None:
    """The page must derive ``wss://`` from the page protocol, not hardcode ``ws://``.

    Hardcoding ``ws://`` breaks every deployment behind TLS, because mixed content stops
    the socket from opening at all.
    """
    with TestClient(app) as client:
        html = client.get("/").text
    assert "wss://" in html, "the UI never derives a wss:// URL, so TLS deployments break"
    assert "ws://" in html
    assert 'location.protocol === "https:"' in html
    assert "new WebSocket(scheme" in html


def test_index_never_hardcodes_a_ws_only_url() -> None:
    """Guard the raw prefix: ``"ws://" + host`` with no scheme check is the bug shape."""
    with TestClient(app) as client:
        html = client.get("/").text
    assert 'new WebSocket("ws://' not in html
    assert "?token=" in html, "the UI cannot pass the API token to the websocket handshake"


def test_index_is_public_even_when_a_token_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_WEB_TOKEN", TOKEN)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing-goal"),
        pytest.param({"goal": ""}, id="empty-goal"),
        pytest.param({"goal": "   "}, id="whitespace-goal"),
        pytest.param({"goal": "x" * 4_001}, id="oversized-goal"),
        pytest.param({"goal": "ok", "max_cycles": 0}, id="max-cycles-zero"),
        pytest.param({"goal": "ok", "max_cycles": 10_001}, id="max-cycles-too-large"),
        pytest.param({"goal": "ok", "max_cycles": "lots"}, id="max-cycles-not-an-int"),
    ],
)
def test_post_run_rejects_bad_bodies(payload: dict[str, Any]) -> None:
    """Validation failures are 400, never FastAPI's default 422."""
    with TestClient(app) as client:
        response = client.post("/api/runs", json=payload)
    assert response.status_code == 400, response.text
    assert response.json()["detail"]


def test_post_run_returns_202_with_a_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/runs",
                json={"goal": "ship it", "model": "fake-model", "max_cycles": 3},
            )
            assert response.status_code == 202, response.text
            body = response.json()
            assert body["run_id"]
            assert body["goal"] == "ship it"
            assert isinstance(body["started_at"], float)
            assert len(created) == 1
            assert created[0].settings.max_cycles == 3
            assert created[0].settings.model == "fake-model"
            assert registry.get(body["run_id"]) is not None
    finally:
        gate.open()


def test_post_run_trims_the_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            body = client.post("/api/runs", json={"goal": "  padded goal  "}).json()
        assert body["goal"] == "padded goal"
        assert created[0].goal == "padded goal"
    finally:
        gate.open()


def test_post_run_refuses_to_exceed_the_concurrency_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            accepted = [
                client.post("/api/runs", json={"goal": f"goal {index}"})
                for index in range(MAX_ACTIVE_RUNS)
            ]
            assert [response.status_code for response in accepted] == [202] * MAX_ACTIVE_RUNS
            rejected = client.post("/api/runs", json={"goal": "one too many"})
            assert rejected.status_code == 429, rejected.text
            assert "too many runs" in rejected.json()["detail"]
        assert len(created) == MAX_ACTIVE_RUNS
    finally:
        gate.open()


def test_get_runs_lists_runs_newest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            first = client.post("/api/runs", json={"goal": "older"}).json()["run_id"]
            second = client.post("/api/runs", json={"goal": "newer"}).json()["run_id"]
            response = client.get("/api/runs")
        assert response.status_code == 200
        records = response.json()
        assert isinstance(records, list), records
        assert [record["id"] for record in records] == [second, first]
        assert all(record["running"] is True for record in records)
        assert all(record["halted_reason"] is None for record in records)
        assert {record["goal"] for record in records} == {"older", "newer"}
        # The overview mirrors whatever the orchestrator currently reports.
        assert wait_until(lambda: all(fake.cycles == 1 for fake in created))
        assert {record["cycles"] for record in records} == {1}
    finally:
        gate.open()


def test_get_run_detail_is_json_serialisable(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _Gate()
    install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            run_id = client.post("/api/runs", json={"goal": "detail"}).json()["run_id"]
            response = client.get(f"/api/runs/{run_id}")
            assert response.status_code == 200
            body = response.json()
            assert json.loads(json.dumps(body)) == body
            detail = client.get(f"/api/runs/{run_id}")
    finally:
        gate.open()

    assert body["id"] == run_id
    assert body["goal"] == "detail"
    assert body["tasks"], body
    assert body["tasks"][0]["id"] == "1"
    assert body["summary"] is None
    assert body["error"] is None
    assert isinstance(body["started_at"], float)
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 3, "requests": 2}
    assert body["events"]
    for event in body["events"]:
        assert_event_frame(dict(event))
    # Repeated reads are stable: the detail endpoint must not mutate stored state.
    assert detail.status_code == 200
    assert detail.json()["id"] == run_id


def test_get_unknown_run_is_404() -> None:
    with TestClient(app) as client:
        response = client.get("/api/runs/nope")
        assert response.status_code == 404
        assert "unknown run id" in response.json()["detail"]


def test_stop_unknown_run_is_404() -> None:
    with TestClient(app) as client:
        response = client.post("/api/runs/nope/stop")
    assert response.status_code == 404
    assert "unknown run id" in response.json()["detail"]


def test_stop_reaches_the_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            run_id = client.post("/api/runs", json={"goal": "stoppable"}).json()["run_id"]
            response = client.post(f"/api/runs/{run_id}/stop")
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["run_id"] == run_id
            assert body["stopped"] is True
            assert body["halted_reason"] == "stopped"
    finally:
        gate.open()
    assert len(created) == 1
    assert created[0].stop_calls == 1


def test_run_events_support_since_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _Gate()
    install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            run_id = client.post("/api/runs", json={"goal": "events"}).json()["run_id"]
            all_events = client.get(f"/api/runs/{run_id}/events").json()
            assert all_events["run_id"] == run_id
            assert all_events["running"] is True
            assert all_events["last_timestamp"] > 0
            every = all_events["events"]
            assert every

            watermark = every[0]["timestamp"]
            fresh = client.get(
                f"/api/runs/{run_id}/events", params={"since": watermark}
            ).json()["events"]
            assert all(event["timestamp"] > watermark for event in fresh)
            assert len(fresh) == len(every) - 1

            capped = client.get(f"/api/runs/{run_id}/events", params={"limit": 2}).json()
            assert len(capped["events"]) == 2

            empty = client.get(f"/api/runs/{run_id}/events", params={"limit": 0}).json()
            assert empty["events"] == []
            assert empty["last_timestamp"] == all_events["last_timestamp"]
    finally:
        gate.open()


def test_run_events_for_an_unknown_run_is_404() -> None:
    with TestClient(app) as client:
        assert client.get("/api/runs/nope/events").status_code == 404


# ---------------------------------------------------------------------------
# authentication
# ---------------------------------------------------------------------------


def test_api_is_open_when_no_token_is_configured() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["auth_required"] is False
        assert client.get("/api/runs").status_code == 200


def test_api_requires_a_bearer_token_when_one_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_WEB_TOKEN", TOKEN)
    install_fake(monkeypatch)
    with TestClient(app) as client:
        missing = client.get("/api/health")
        assert missing.status_code == 401
        assert missing.headers["www-authenticate"] == "Bearer"
        assert "FURROW_WEB_TOKEN" in missing.json()["detail"]

        wrong = client.get("/api/health", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401

        wrong_scheme = client.get("/api/health", headers={"Authorization": f"Basic {TOKEN}"})
        assert wrong_scheme.status_code == 401

        authed = {"Authorization": f"Bearer {TOKEN}"}
        assert client.get("/api/runs", headers=authed).status_code == 200

        good = client.get("/api/health", headers=authed)
        assert good.status_code == 200
        assert good.json()["auth_required"] is True

        # Starting a run is privileged too.
        assert client.post("/api/runs", json={"goal": "nope"}).status_code == 401


def test_blank_token_disables_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_WEB_TOKEN", "   ")
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["auth_required"] is False


BLANK_TOKENS = ("", " ", "   ", "\t", "\n", " \t\n ")


@pytest.mark.parametrize("blank", BLANK_TOKENS)
def test_a_blank_token_is_reported_as_no_token_at_all(
    blank: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3-R2: a blank token means no token, so the docs come back open too.

    `configured_token()` strips the value and treats an empty result as auth-disabled.
    That is the right behaviour - a token of spaces cannot authenticate anything - but it
    means a stray trailing space in a systemd unit or a `.env` line silently re-opens
    `/docs`, `/redoc` and `/openapi.json` on a deployment the operator believed was
    protected. Pinning it here so the behaviour cannot drift without a doc change.
    """
    monkeypatch.setenv("FURROW_WEB_TOKEN", blank)
    assert server_module.configured_token() is None
    with TestClient(app) as client:
        assert client.get("/api/health").json()["auth_required"] is False
        assert client.get("/api/runs").status_code == 200
        for route in DOCS_ROUTES:
            assert client.get(route).status_code == 200, route


@pytest.mark.parametrize("blank", BLANK_TOKENS)
def test_a_blank_token_logs_a_loud_startup_warning(
    blank: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The gap this closes: a set-but-blank token is invisible.

    `configured_token()` answers `None`, so `log_security_notice()` used to fall into the
    generic "serving with NO authentication" branch - a warning the operator has already
    learned to ignore from the default loopback run, and one that never says the variable
    they set was the problem. The variable IS set here, so the operator believes the gate
    is closed. It must be called out by name, at ERROR level.
    """
    monkeypatch.setenv("FURROW_WEB_TOKEN", blank)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=server_module.logger.name):
        server_module.log_security_notice("0.0.0.0")
    records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    notice = records[0].getMessage()
    assert "FURROW_WEB_TOKEN" in notice
    assert "BLANK" in notice
    assert "DISABLED" in notice
    assert "0.0.0.0" in notice


def test_a_blank_token_does_not_also_emit_the_generic_unauthenticated_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One specific ERROR beats two overlapping warnings; the generic one would be noise."""
    monkeypatch.setenv("FURROW_WEB_TOKEN", " ")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=server_module.logger.name):
        server_module.log_security_notice()
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(messages) == 1
    assert "NO authentication" not in messages[0]


@pytest.mark.parametrize("value,expected", [("real-token", True), (" tok ", True)])
def test_a_real_token_takes_the_quiet_path_and_never_the_blank_warning(
    value: str, expected: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The warning must not fire on the two configurations that are working as intended."""
    monkeypatch.setenv("FURROW_WEB_TOKEN", value)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=server_module.logger.name):
        server_module.log_security_notice()
    assert (server_module.configured_token() is not None) is expected
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_an_unset_token_keeps_only_the_generic_unauthenticated_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The zero-config loopback path is unchanged: one WARNING, and it is the old one."""
    monkeypatch.delenv("FURROW_WEB_TOKEN", raising=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=server_module.logger.name):
        server_module.log_security_notice()
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "NO authentication" in records[0].getMessage()


def test_the_blank_token_warning_also_fires_on_the_real_startup_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`log_security_notice` is reached from the app lifespan, so ASGI startup must log it."""
    monkeypatch.setenv("FURROW_WEB_TOKEN", "   ")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=server_module.logger.name):
        with TestClient(app):
            pass
    assert any(
        "FURROW_WEB_TOKEN" in record.getMessage() and record.levelno >= logging.ERROR
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


# ---------------------------------------------------------------------------
# docs surface (F3: must not be the way around the token gate)
# ---------------------------------------------------------------------------

DOCS_ROUTES = ("/docs", "/redoc", "/openapi.json")


@pytest.mark.parametrize("route", DOCS_ROUTES)
def test_docs_are_served_when_no_token_is_configured(route: str) -> None:
    """Zero-config loopback keeps the browsable docs and the schema."""
    with TestClient(app) as client:
        response = client.get(route)
    assert response.status_code == 200, response.text


def test_the_openapi_schema_describes_the_api_when_no_token_is_configured() -> None:
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Furrow"
    assert "/api/health" in schema["paths"]
    assert "/api/runs" in schema["paths"]


@pytest.mark.parametrize("route", DOCS_ROUTES)
def test_docs_reject_an_anonymous_reader_when_a_token_is_set(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap: these routes used to be registered on the app with no auth dependency."""
    monkeypatch.setenv("FURROW_WEB_TOKEN", TOKEN)
    with TestClient(app) as client:
        response = client.get(route)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("route", DOCS_ROUTES)
def test_docs_are_not_served_at_all_once_a_token_is_set(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a valid credential gets 404: the browser cannot send the header Swagger needs.

    Gating ``/openapi.json`` behind the bearer token alone would leave ``/docs`` rendering
    as a broken "failed to load API definition" page, because Swagger UI fetches the schema
    from the same origin with no ``Authorization`` header. The docs are therefore published
    only in the unauthenticated loopback configuration.
    """
    monkeypatch.setenv("FURROW_WEB_TOKEN", TOKEN)
    with TestClient(app) as client:
        wrong = client.get(route, headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        authed = client.get(route, headers={"Authorization": f"Bearer {TOKEN}"})
    assert authed.status_code == 404
    assert "FURROW_WEB_TOKEN" in authed.json()["detail"]


def test_the_api_itself_still_works_with_a_token_while_the_docs_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the docs must not close the routes they used to document."""
    monkeypatch.setenv("FURROW_WEB_TOKEN", TOKEN)
    install_fake(monkeypatch)
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 401
        authed = {"Authorization": f"Bearer {TOKEN}"}
        assert client.get("/api/health", headers=authed).status_code == 200
        assert client.get("/api/runs", headers=authed).status_code == 200



def test_websocket_requires_the_token_query_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_WEB_TOKEN", TOKEN)
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            for path in ("/ws", "/ws?token=wrong"):
                with pytest.raises(WebSocketDisconnect) as refused:
                    with client.websocket_connect(path) as websocket:
                        websocket.send_json({"goal": "denied"})
                        receive_frame(websocket)
                assert refused.value.code == 1008, path
                assert created == [], path

            # The handshake is accepted, and the query parameter is what authorises it.
            with client.websocket_connect(f"/ws?token={TOKEN}") as websocket:
                websocket.send_json({"goal": "allowed"})
                started = receive_frame(websocket)
                assert started["kind"] == "control"
                assert started["type"] == "started"
                gate.open()
                read_until_done(websocket)
    finally:
        gate.open()
    assert len(created) == 1


# ---------------------------------------------------------------------------
# resource hygiene
# ---------------------------------------------------------------------------


def test_websocket_session_leaves_no_subscribers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finished stream must release its bus subscriber and let the run finish cleanly."""
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            _, stream = start_over_websocket(client, created, gate)
            fake = created[0]
            run = registry.get(stream[0]["run_id"])
            assert run is not None
            assert wait_until(lambda: fake.completed)
            assert wait_until(lambda: run.task is not None and run.task.done())
            assert wait_until(lambda: run.summary is not None)
            assert len(fake.bus._subscribers) == 0
            assert run.summary is not None
            assert run.summary.halted_reason == "goal_complete"
            assert run.running is False
    finally:
        gate.open()
    assert len(created[0].bus._subscribers) == 0


def test_early_disconnect_leaves_no_subscribers_and_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaving mid-stream must not raise server-side, leak a subscriber or orphan the run."""
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"goal": "leaving early"})
                started = receive_frame(websocket)
                read_frames(websocket, len(FakeOrchestrator.PRE_GATE_EVENTS))
                run_id = started["run_id"]
            # The socket is gone. No exception was raised on either side of the wire.
            fake = created[0]
            run = registry.get(run_id)
            assert run is not None
            assert wait_until(lambda: len(fake.bus._subscribers) == 0)
            # The run itself is untouched and still finishes on its own.
            assert fake.completed is False
            gate.open()
            assert wait_until(lambda: fake.completed)
            assert wait_until(lambda: run.summary is not None)
            assert len(fake.bus._subscribers) == 0
    finally:
        gate.open()


def test_stop_after_a_disconnect_still_reaches_the_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping the socket does not make the run unstoppable."""
    gate = _Gate()
    created = install_fake(monkeypatch, gate=gate)
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"goal": "detach then stop"})
                started = receive_frame(websocket)
                read_frames(websocket, len(FakeOrchestrator.PRE_GATE_EVENTS))
                run_id = started["run_id"]
            response = client.post(f"/api/runs/{run_id}/stop")
            assert response.status_code == 200
            assert response.json()["stopped"] is True
            assert created[0].stop_calls == 1
            assert wait_until(lambda: created[0].completed)
    finally:
        gate.open()
