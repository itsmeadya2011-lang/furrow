"""Tests for the FastAPI monitoring UI: ``furrow/web/server.py``.

Everything is driven through :class:`fastapi.testclient.TestClient` -- the HTTP routes, the
WebSocket handshake and the frame stream -- because the properties that matter here are
about *what a browser actually receives*: which frames arrive, in what order, and with
what JSON.

``Orchestrator`` is replaced by a fake that replays a scripted event sequence through the
``on_event`` callback it was handed and returns a hand-built :class:`RunSummary`. The
script is emitted through the **real** :class:`~furrow.core.events.EventEmitter`, not
hand-built dicts, so the frames on the wire are produced by the production code path --
which is what makes these tests capable of catching the ``"type": "EventType.RUN_START"``
regression (:func:`test_websocket_receives_one_frame_per_event_with_plain_string_types`).

Two other properties get the same treatment:

* the run must be a **background task**. ``_launch`` uses :func:`asyncio.create_task` and
  the socket is fed by a separate ``_pump`` task reading an :class:`asyncio.Queue`; the
  pre-fix handler awaited the whole plan/worker/tester loop inline, which pinned the
  connection open for the entire run and made the UI show nothing until the agent was
  done. :func:`test_the_first_frame_arrives_before_the_run_finishes` is the regression.
* the module-level ``_active_runs`` registry is process-wide state. An autouse fixture
  clears it at both ends of every test, and
  :func:`test_the_run_registry_leaves_every_test_clean` (last in the file, so it runs after
  every other test in the session) proves nothing leaked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import furrow.web.server as server
from furrow.core import EventEmitter, EventType, RunSummary
from furrow.web.server import MAX_CONCURRENT_RUNS

#: A scripted event sequence mirroring the real orchestrator's payload contract: ``type``
#: is the plain member value, and every event carries ``cycle``.
SCRIPT_EVENTS: list[dict[str, Any]] = [
    {
        "type": "run_start",
        "cycle": 0,
        "goal": "fake goal",
        "workspace": "/tmp/ws",
        "provider": "anthropic",
        "models": {"planner": "m", "worker": "m", "tester": "m"},
    },
    {"type": "cycle_start", "cycle": 1},
    {
        "type": "plan",
        "cycle": 1,
        "rationale": "do the thing",
        "task_count": 1,
        "tasks": [
            {
                "id": "t1",
                "description": "write it",
                "files": ["a.py"],
                "dependencies": [],
                "status": "pending",
            }
        ],
    },
    {"type": "task_start", "cycle": 1, "task_id": "t1", "description": "write it"},
    {
        "type": "task_end",
        "cycle": 1,
        "task_id": "t1",
        "description": "write it",
        "status": "completed",
        "written_files": ["a.py"],
        "result": "wrote a.py",
    },
    {
        "type": "test",
        "cycle": 1,
        "passed": True,
        "skipped": False,
        "summary": "1 passed",
        "failures": [],
    },
    {
        "type": "cycle_end",
        "cycle": 1,
        "completed": 1,
        "failed": 0,
        "skipped": 0,
        "passed": True,
    },
    {
        "type": "run_end",
        "cycle": 1,
        "cycles": 1,
        "completed": 1,
        "failed": 0,
        "skipped": 0,
        "stopped_because": "tests-passed",
    },
]

#: The scripted ``type`` values, in order.
SCRIPT_TYPES = [event["type"] for event in SCRIPT_EVENTS]


def make_summary(goal: str = "fake goal", stopped_because: str = "tests-passed") -> RunSummary:
    """Hand-built :class:`RunSummary`, exactly as the real ``run()`` would return one."""
    return RunSummary(
        goal=goal,
        cycles=1,
        completed=1,
        failed=0,
        skipped=0,
        stopped_because=stopped_because,
        tasks=[],
    )


class FakeOrchestrator:
    """Stand-in for :class:`furrow.core.Orchestrator`.

    Mirrors the real constructor signature, replays :data:`SCRIPT_EVENTS` through the
    ``on_event`` callback it was given, and returns a hand-built :class:`RunSummary`.
    Class attributes let a test slow it down or make it explode without a bespoke class.
    """

    #: Seconds slept between scripted events -- how a test makes a run "slow".
    delay: float = 0.0
    #: When set, :meth:`run` raises it *after* emitting the whole script.
    raise_exc: BaseException | None = None
    #: ``{"goal": ..., "settings": ...}`` per construction.
    seen: list[dict[str, Any]] = []
    #: Every instance built during the current test.
    instances: list[FakeOrchestrator] = []

    def __init__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        console: Any = None,
    ) -> None:
        """See class docstring."""
        self.goal = goal
        self.settings = settings
        self.on_event = on_event
        self.console = console
        self.sent: list[dict[str, Any]] = []
        type(self).seen.append({"goal": goal, "settings": settings})
        type(self).instances.append(self)

    async def run(self) -> RunSummary:
        """Emit the script, then return a summary (or raise, if configured to)."""
        emitter = EventEmitter(self.on_event)
        for scripted in SCRIPT_EVENTS:
            self.sent.append(scripted)
            fields = {key: value for key, value in scripted.items() if key != "type"}
            emitter.emit(EventType(scripted["type"]), **fields)
            await asyncio.sleep(type(self).delay)
        exc: BaseException | None = type(self).raise_exc
        if exc is not None:
            raise exc
        return make_summary(goal=self.goal)


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    """Every test starts and ends with an empty process-wide run registry."""
    server._active_runs.clear()
    yield
    server._active_runs.clear()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> type[FakeOrchestrator]:
    """Install :class:`FakeOrchestrator` as ``furrow.web.server.Orchestrator``."""
    FakeOrchestrator.delay = 0.0
    FakeOrchestrator.raise_exc = None
    FakeOrchestrator.seen = []
    FakeOrchestrator.instances = []
    monkeypatch.setattr(server, "Orchestrator", FakeOrchestrator)
    return FakeOrchestrator


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A :class:`TestClient` used as a context manager.

    The ``with`` is load-bearing: starlette builds a fresh event loop per request when
    there is none, and a background task created by one request would be torn down with
    it and never progress.
    """
    with TestClient(server.app) as test_client:
        yield test_client


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    """Poll ``predicate`` until it is true or ``timeout`` expires."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --------------------------------------------------------------------------- #
# GET / -- the self-contained UI
# --------------------------------------------------------------------------- #


def test_index_serves_self_contained_ui(client: TestClient) -> None:
    """The page must derive its socket URL from the page origin, not hardcode one."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert html == server.INDEX_HTML
    # ws vs wss is chosen from the page's own protocol, so the same file works behind TLS.
    assert "location.protocol" in html
    assert "'wss'" in html and "'ws'" in html
    assert "new WebSocket(scheme + '://' + location.host + '/ws')" in html
    assert "ws://" not in html
    assert "wss://" not in html
    # An onopen handler exists: the goal is only sent once the socket is actually open.
    assert "ws.onopen" in html
    assert "ws.onmessage" in html
    assert "ws.onclose" in html
    # The buffer is bounded and reassigned wholesale, never grown with `+=` forever.
    assert "textContent +=" not in html
    assert "out.textContent = lines.join" in html
    assert "MAX_LINES" in html
    assert "lines.splice(" in html
    # Every event type the orchestrator can emit has a formatter branch.
    for etype in SCRIPT_TYPES + ["error"]:
        assert f"case '{etype}'" in html
    assert "stopped_because" in html
    # A second submit while a run is in flight is refused.
    assert "if (running) { return; }" in html


# --------------------------------------------------------------------------- #
# GET /api/health
# --------------------------------------------------------------------------- #


def test_health_reports_status_runs_and_version(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """The probe carries exactly three keys and counts in-flight runs."""
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "runs", "version"}
    assert body["status"] == "ok"
    assert body["runs"] == 0
    assert body["version"] == server.__version__

    fake.delay = 0.5  # keep the run in flight while we look at the count
    client.post("/api/runs", json={"goal": "g"})
    assert wait_until(lambda: len(fake.seen) == 1)
    assert client.get("/api/health").json()["runs"] == 1


# --------------------------------------------------------------------------- #
# POST /api/runs -- validation
# --------------------------------------------------------------------------- #


def test_create_run_returns_an_id(client: TestClient, fake: type[FakeOrchestrator]) -> None:
    """A well-formed body is accepted and the goal is whitespace-trimmed."""
    ok = client.post("/api/runs", json={"goal": "  padded  "})
    assert ok.status_code == 200
    body = ok.json()
    assert set(body) == {"id", "goal"}
    assert body["goal"] == "padded"
    assert isinstance(body["id"], str)
    assert body["id"]


def test_create_run_rejects_a_blank_goal(client: TestClient, fake: type[FakeOrchestrator]) -> None:
    """Whitespace-only and empty goals are 422, naming the offending field."""
    blank = client.post("/api/runs", json={"goal": "   "})
    assert blank.status_code == 422, blank.text
    assert blank.json()["detail"][0]["loc"] == ["body", "goal"]


def test_create_run_rejects_an_empty_object(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """``{}`` has no goal at all: 422, not a 500."""
    missing = client.post("/api/runs", json={})
    assert missing.status_code == 422, missing.text


def test_create_run_rejects_a_json_array_body(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """A JSON array must be a 422, never an ``AttributeError``-driven 500.

    The pre-fix handler took the raw body as ``Any`` and called ``data.get(...)`` on it, so
    a well-formed-but-wrong-shaped request crashed with a stack trace in the server log.
    """
    array_body = client.post("/api/runs", json=[1, 2, 3])
    assert array_body.status_code == 422, array_body.text
    assert "AttributeError" not in array_body.text
    assert server._active_runs == {}, "a rejected request must not create a run record"


def test_create_run_validation_happens_before_any_run_is_started(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """A batch of bad payloads starts nothing at all."""
    for payload in ({"goal": ""}, {"goal": "   "}, {}, [1, 2, 3], "a string", {"goal": 5}):
        assert client.post("/api/runs", json=payload).status_code == 422, payload
    assert fake.seen == []
    assert server._active_runs == {}


def test_websocket_rejects_bad_payloads_with_error_frames(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """Every unusable handshake payload produces a readable error frame, not a crash."""
    for payload, needle in (
        ("[1, 2, 3]", "payload"),
        ('"just a string"', "payload"),
        ("not json at all", "not valid JSON"),
        ('{"goal": "  "}', "goal"),
        ("{}", "goal"),
    ):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(payload)
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "error", (payload, frame)
            assert needle in frame["message"], (payload, frame)
            assert frame["cycle"] == 0
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
    assert server._active_runs == {}, "no run may be created from a rejected payload"
    assert fake.seen == []


# --------------------------------------------------------------------------- #
# /ws -- the frame stream
# --------------------------------------------------------------------------- #


def test_websocket_receives_one_frame_per_event_with_plain_string_types(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """One text frame per emitted event, in order, with wire-value ``type`` strings.

    The regression this pins: the handler accepted the socket, ran the orchestrator and
    **never called any ``send_*`` method**, so the browser received zero frames and the UI
    displayed nothing at all. And because the ``type`` field used to carry
    ``str(EventType.X)``, even a delivered frame would not have matched the UI's
    ``case 'run_end':`` branches.
    """
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "fake goal"})
        frames = [json.loads(ws.receive_text()) for _ in range(len(SCRIPT_EVENTS))]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()

    assert len(frames) == len(SCRIPT_EVENTS)
    assert [frame["type"] for frame in frames] == SCRIPT_TYPES
    for frame in frames:
        assert isinstance(frame["type"], str)
        assert "EventType" not in frame["type"]
        assert isinstance(frame["cycle"], int)
    assert frames == SCRIPT_EVENTS
    # The run really emitted them all -- the frames are not synthesised by the web layer.
    assert len(fake.instances[0].sent) == len(SCRIPT_EVENTS)


def test_websocket_stream_is_recorded_against_its_run(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """Every streamed frame is also retained, so ``GET /api/runs/{id}`` can replay it."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "fake goal"})
        for _ in range(len(SCRIPT_EVENTS)):
            ws.receive_text()

    run_id = next(iter(server._active_runs))
    body = client.get(f"/api/runs/{run_id}").json()
    assert body["history"] == SCRIPT_EVENTS
    assert body["events"] == len(SCRIPT_EVENTS)
    assert body["done"] is True
    assert body["cancelled"] is False
    assert body["error"] is None
    assert body["summary"] == {
        "goal": "fake goal",
        "cycles": 1,
        "completed": 1,
        "failed": 0,
        "skipped": 0,
        "stopped_because": "tests-passed",
        "tasks": [],
    }


def test_a_raising_orchestrator_ends_the_stream_with_an_error_frame(
    client: TestClient, fake: type[FakeOrchestrator], caplog: pytest.LogCaptureFixture
) -> None:
    """A failed run still gets its scripted frames, then a terminal ``error`` frame.

    The pre-fix handler let the exception escape, so the socket just closed and the user
    never learned why. Now the failure is logged with a traceback *and* pushed on the
    wire.
    """
    fake.raise_exc = ValueError("planner returned no tasks")
    with (
        caplog.at_level(logging.ERROR, logger="furrow.web"),
        client.websocket_connect("/ws") as ws,
    ):
        ws.send_json({"goal": "fake goal"})
        frames = [json.loads(ws.receive_text()) for _ in range(len(SCRIPT_EVENTS) + 1)]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()

    last = frames[-1]
    assert last["type"] == "error"
    assert "ValueError: planner returned no tasks" in last["message"]
    assert [frame["type"] for frame in frames[:-1]] == SCRIPT_TYPES
    assert "furrow run" in caplog.text
    assert "Traceback" in caplog.text

    record = next(iter(server._active_runs.values()))
    assert record.done is True
    assert record.error == "ValueError: planner returned no tasks"
    assert record.summary is None
    assert record.history[-1]["type"] == "error"


def test_a_raising_orchestrator_over_http_is_recorded_not_raised(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """``POST /api/runs`` has already returned 200; the failure lands on the record."""
    fake.raise_exc = RuntimeError("no API key configured")
    response = client.post("/api/runs", json={"goal": "g"})
    assert response.status_code == 200
    run_id = response.json()["id"]

    assert wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("error") is not None
    ), "the run never recorded a terminal error"
    body = client.get(f"/api/runs/{run_id}").json()
    assert body["error"] == "RuntimeError: no API key configured"
    assert body["done"] is True
    assert body["summary"] is None
    assert body["history"][-1]["type"] == "error"


def test_model_and_workspace_overrides_reach_the_orchestrator(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """Per-run overrides build run-scoped settings without touching the singleton."""
    response = client.post(
        "/api/runs",
        json={"goal": "g", "model": "m-x", "workspace": "/tmp/ws-override"},
    )
    assert response.status_code == 200
    assert wait_until(lambda: len(fake.seen) == 1)

    seen = fake.seen[0]
    assert seen["goal"] == "g"
    assert seen["settings"] is not None, "an override must produce run-scoped settings"
    assert seen["settings"].planner_model == "m-x"
    assert seen["settings"].worker_model == "m-x"
    assert seen["settings"].tester_model == "m-x"
    assert str(seen["settings"].workspace) == "/tmp/ws-override"
    assert seen["settings"] is not server.get_settings()
    assert server.get_settings().planner_model != "m-x"


def test_a_run_without_overrides_gets_the_untouched_singleton(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """The common path must not deep-copy the settings for every run."""
    client.post("/api/runs", json={"goal": "g"})
    assert wait_until(lambda: len(fake.seen) == 1)
    assert fake.seen[0]["settings"] is None


# --------------------------------------------------------------------------- #
# POST /api/runs/{id}/stop
# --------------------------------------------------------------------------- #


def test_stop_cancels_a_long_running_run(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """Stopping a run really cancels its task, it does not merely set a flag."""
    fake.delay = 0.2
    run_id = client.post("/api/runs", json={"goal": "slow"}).json()["id"]
    assert wait_until(lambda: len(server._active_runs[run_id].history) >= 1)
    assert server._active_runs[run_id].done is False

    stopped = client.post(f"/api/runs/{run_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json() == {"id": run_id, "cancelled": True}

    assert wait_until(lambda: client.get(f"/api/runs/{run_id}").json()["done"] is True)
    body = client.get(f"/api/runs/{run_id}").json()
    assert body["cancelled"] is True
    assert body["done"] is True
    assert body["summary"] is None
    task_handle = server._active_runs[run_id].task
    assert task_handle is not None
    assert task_handle.cancelled() is True
    assert len(body["history"]) < len(SCRIPT_EVENTS), "the run should have been cut short"


def test_stop_is_idempotent_and_unknown_ids_are_404(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """Stopping twice is harmless; stopping an unknown id is a clean 404."""
    run_id = client.post("/api/runs", json={"goal": "g"}).json()["id"]
    first = client.post(f"/api/runs/{run_id}/stop")
    second = client.post(f"/api/runs/{run_id}/stop")
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == {"id": run_id, "cancelled": True}

    missing = client.post("/api/runs/does-not-exist/stop")
    assert missing.status_code == 404
    assert "detail" in missing.json()


def test_a_client_disconnect_cancels_the_run(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """Closing the socket is a stop request: nobody is left watching the run."""
    fake.delay = 0.2
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "slow"})
        assert json.loads(ws.receive_text())["type"] == "run_start"
        run_id = next(iter(server._active_runs))

    assert wait_until(lambda: _task_of(run_id).done())
    assert server._active_runs[run_id].cancelled is True


def _task_of(run_id: str) -> asyncio.Task[None]:
    """The background task behind `run_id`; fails loudly if the record lost it."""
    task = server._active_runs[run_id].task
    assert task is not None, f"run {run_id!r} has no background task"
    return task


# --------------------------------------------------------------------------- #
# GET /api/runs and GET /api/runs/{id}
# --------------------------------------------------------------------------- #


def test_list_runs_shape_and_order(client: TestClient, fake: type[FakeOrchestrator]) -> None:
    """The collection route lists known runs oldest first with the shared summary shape."""
    assert client.get("/api/runs").json() == []
    first = client.post("/api/runs", json={"goal": "one"}).json()["id"]
    second = client.post("/api/runs", json={"goal": "two"}).json()["id"]
    assert wait_until(lambda: all(r.done for r in server._active_runs.values()))

    items = client.get("/api/runs").json()
    assert isinstance(items, list)
    assert [item["id"] for item in items] == [first, second]
    for item in items:
        assert set(item) == {"id", "goal", "started_at", "done", "cancelled", "events"}
        assert isinstance(item["started_at"], (int, float))
        assert item["done"] is True
        assert item["cancelled"] is False
        assert item["events"] == len(SCRIPT_EVENTS)
    assert items[0]["goal"] == "one"


def test_get_one_run_adds_elapsed_error_history_and_summary(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """The detail route is the list item plus the four run-specific fields."""
    run_id = client.post("/api/runs", json={"goal": "one"}).json()["id"]
    assert wait_until(lambda: client.get(f"/api/runs/{run_id}").json()["done"] is True)

    body = client.get(f"/api/runs/{run_id}").json()
    assert set(body) == {
        "id",
        "goal",
        "started_at",
        "done",
        "cancelled",
        "events",
        "elapsed",
        "error",
        "history",
        "summary",
    }
    assert isinstance(body["elapsed"], (int, float))
    assert body["error"] is None
    assert body["summary"]["stopped_because"] == "tests-passed"

    assert client.get("/api/runs/nope").status_code == 404
    assert "detail" in client.get("/api/runs/nope").json()


# --------------------------------------------------------------------------- #
# The concurrency cap
# --------------------------------------------------------------------------- #


def test_the_concurrency_cap_accepts_exactly_max_concurrent_runs(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """``MAX_CONCURRENT_RUNS`` may be in flight; the next request is refused with 429."""
    assert MAX_CONCURRENT_RUNS == 4
    fake.delay = 1.0
    accepted = []
    for index in range(MAX_CONCURRENT_RUNS):
        response = client.post("/api/runs", json={"goal": f"g{index}"})
        assert response.status_code == 200, response.text
        accepted.append(response.json()["id"])
    assert wait_until(
        lambda: client.get("/api/health").json()["runs"] == MAX_CONCURRENT_RUNS
    )

    overflow = client.post("/api/runs", json={"goal": "too many"})
    assert overflow.status_code == 429
    assert "in flight" in overflow.json()["detail"]
    assert set(server._active_runs) == set(accepted), "a refused run leaves no record"


def test_the_concurrency_cap_also_guards_the_websocket_handshake(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """The WS path applies the same cap, so a browser tab cannot bypass it."""
    fake.delay = 1.0
    for _ in range(MAX_CONCURRENT_RUNS):
        client.post("/api/runs", json={"goal": "busy"})
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "one too many"})
        frame = json.loads(ws.receive_text())
        assert frame["type"] == "error"
        assert "in flight" in frame["message"]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


# --------------------------------------------------------------------------- #
# run() -- the uvicorn entry point
# --------------------------------------------------------------------------- #


def test_run_defaults_to_loopback_and_forwards_uvicorn_arguments(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Defaults are loopback, the port range is enforced, and uvicorn gets the real args.

    ``127.0.0.1`` and not ``0.0.0.0`` is the security-relevant half: the endpoint is
    unauthenticated and drives an agent with file-write and subprocess access.
    """
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))

    with caplog.at_level(logging.WARNING, logger="furrow.web"):
        server.run()
    args, kwargs = calls[-1]
    assert args[0] is server.app
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000
    assert kwargs["reload"] is False
    assert kwargs["log_level"] == server.get_settings().log_level.lower()
    assert caplog.text == "", "loopback must not warn"


def test_run_warns_loudly_for_a_public_bind(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-loopback host is published to the whole network and says so."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))

    with caplog.at_level(logging.WARNING, logger="furrow.web"):
        server.run(host="0.0.0.0", port=9000)

    assert calls[-1][1]["host"] == "0.0.0.0"
    assert calls[-1][1]["port"] == 9000
    assert "unauthenticated" in caplog.text
    assert "subprocess" in caplog.text


def test_run_passes_an_import_string_when_reloading(monkeypatch: pytest.MonkeyPatch) -> None:
    """uvicorn re-imports its target on reload, so it needs a string rather than an object."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))

    server.run(reload=True)

    assert calls[-1][0][0] == "furrow.web.server:app"
    assert calls[-1][1]["reload"] is True


@pytest.mark.parametrize("port", [0, -1, 65536, 99999])
def test_run_rejects_a_port_outside_the_valid_range(
    monkeypatch: pytest.MonkeyPatch, port: int
) -> None:
    """Ports 0, negative and above 65535 are refused before uvicorn ever starts."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))

    with pytest.raises(ValueError, match="between 1 and 65535"):
        server.run(port=port)
    assert calls == []


@pytest.mark.parametrize("port", [1, 8000, 65535])
def test_run_accepts_the_boundary_ports(monkeypatch: pytest.MonkeyPatch, port: int) -> None:
    """1 and 65535 are inclusive bounds."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))

    server.run(port=port)

    assert calls[-1][1]["port"] == port


# --------------------------------------------------------------------------- #
# The run really is a background task
# --------------------------------------------------------------------------- #


def test_the_first_frame_arrives_before_the_run_finishes(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """Frames stream while the run is still going -- the regression for inline execution.

    The pre-fix handler awaited the whole loop before reading anything off the socket, so
    the browser saw one frame when the agent was already done (or, for a long run, nothing
    at all). Here the fake sleeps between events, so an inline handler could not have
    produced the first two frames in under a second.
    """
    fake.delay = 0.4
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "slow"})
        started = time.perf_counter()
        first = json.loads(ws.receive_text())
        first_at = time.perf_counter() - started
        second = json.loads(ws.receive_text())
        second_at = time.perf_counter() - started
        run_id = next(iter(server._active_runs))

        # The whole run takes 7 x 0.4s == ~2.8s, so anything under half that is proof
        # the handler did not await the loop before reading the socket. The thresholds are
        # deliberately loose (well clear of a slow CI box) but still an order of magnitude
        # below the inline case.
        assert first["type"] == "run_start"
        assert second["type"] == "cycle_start"
        assert first_at < 1.0, f"first frame took {first_at:.2f}s of a ~2.8s run"
        assert second_at < 1.5, f"second frame took {second_at:.2f}s of a ~2.8s run"
        assert server._active_runs[run_id].done is False


def test_create_run_returns_in_milliseconds_while_the_run_is_in_flight(
    client: TestClient, fake: type[FakeOrchestrator]
) -> None:
    """``POST /api/runs`` acknowledges immediately and the server stays responsive."""
    fake.delay = 0.3  # 7 events x 0.3s == ~2.1s of orchestrator work
    started = time.perf_counter()
    response = client.post("/api/runs", json={"goal": "slow"})
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 1.0, f"POST /api/runs blocked for {elapsed:.2f}s on a ~2.1s run"
    run_id = response.json()["id"]
    assert server._active_runs[run_id].done is False
    assert client.get("/api/health").json()["status"] == "ok"
    assert wait_until(lambda: client.get(f"/api/runs/{run_id}").json()["done"])


# --------------------------------------------------------------------------- #
# Registry internals
# --------------------------------------------------------------------------- #


def test_reap_drops_only_old_finished_records() -> None:
    """A live run is never reaped out from under its own task."""
    finished = server.RunRecord(id="fin", goal="g", started_at=time.monotonic() - 10_000)
    finished.finished_at = time.monotonic() - 10_000
    fresh = server.RunRecord(id="fresh", goal="g", started_at=time.monotonic() - 10_000)
    fresh.finished_at = time.monotonic()
    running = server.RunRecord(id="run", goal="g", started_at=time.monotonic() - 10_000)
    server._active_runs.update({"fin": finished, "fresh": fresh, "run": running})

    server._reap_runs()

    assert set(server._active_runs) == {"fresh", "run"}


def test_history_is_capped_dropping_the_oldest() -> None:
    """A multi-hour run cannot grow the process heap without bound."""
    record = server.RunRecord(id="r", goal="g", started_at=time.monotonic())
    for index in range(server.MAX_HISTORY_PER_RUN + 25):
        server._append_event(record, {"type": "log", "cycle": 0, "n": index})

    assert len(record.history) == server.MAX_HISTORY_PER_RUN
    assert record.history[0]["n"] == 25
    assert record.history[-1]["n"] == server.MAX_HISTORY_PER_RUN + 24


def test_next_run_id_is_unique() -> None:
    ids = {server.next_run_id() for _ in range(500)}
    assert len(ids) == 500
    assert all(len(run_id) == 12 for run_id in ids)


def test_lifespan_cancels_in_flight_runs_on_shutdown(
    fake: type[FakeOrchestrator],
) -> None:
    """Leaving the app context must cancel the run, not orphan its task and subprocesses."""
    fake.delay = 5.0
    with TestClient(server.app) as scoped:
        run_id = scoped.post("/api/runs", json={"goal": "orphan"}).json()["id"]
        assert wait_until(lambda: bool(server._active_runs[run_id].history))

    record = server._active_runs[run_id]
    assert _task_of(run_id).done()
    assert record.cancelled is True
    assert record.finished_at is not None


def test_lifespan_emits_no_deprecation_warning(recwarn: pytest.WarningsRecorder) -> None:
    """The async lifespan must not trip a DeprecationWarning in starlette."""
    with TestClient(server.app) as scoped:
        scoped.get("/api/health")
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_openapi_documents_the_routes(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) >= {
        "/",
        "/api/health",
        "/api/runs",
        "/api/runs/{run_id}",
        "/api/runs/{run_id}/stop",
    }
    assert spec["info"]["title"] == "Furrow"
    assert spec["info"]["version"] == server.__version__
    start = spec["components"]["schemas"]["StartRequest"]["properties"]
    assert set(start) == {"goal", "model", "workspace"}


def test_start_request_validates_directly() -> None:
    """One model guards both entry points; it trims and it rejects the wrong shape."""
    start = server.StartRequest(goal="  padded  ")
    assert start.goal == "padded"
    assert start.model is None
    assert start.workspace is None
    for payload in ({"goal": ""}, {"goal": "  "}, {}, [1], "x", {"goal": 5}):
        with pytest.raises(ValueError):
            server.StartRequest.model_validate(payload)


def test_the_run_registry_leaves_every_test_clean() -> None:
    """Process-wide state check, deliberately defined last in this file.

    Every test above creates runs in the module-level ``_active_runs`` dict; the autouse
    ``clean_registry`` fixture empties it at teardown. If any web test (or anything earlier
    in the session) leaked a record, this fails. It is the same guarantee for the registry
    that ``hermetic_env`` gives the ``get_settings`` lru_cache.
    """
    assert server._active_runs == {}
