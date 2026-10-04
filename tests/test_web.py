"""Contracts for the monitoring web server.

Protected behaviour, one test per contract:

* ``GET /`` serves the UI, and the page derives its websocket scheme from
  ``location.protocol`` so it also works behind TLS. Streamed text reaches the log as
  text only -- the page never builds markup out of server output.
* ``WS /ws`` streams every orchestrator progress event and then exactly one terminal
  frame -- ``result`` for a finished run, ``error`` for one that could not start. This is
  the regression test for the shell that accepted a socket and then never sent anything.
* Bad input (blank goal, non-JSON message) and a raising engine produce an ``error`` frame
  and a clean close, never an unhandled server error.
* A client disconnect -- or an explicit ``{"type": "stop"}`` -- cancels the in-flight run
  task instead of leaking it.
* The REST surface registers a run (``POST /api/start``), lists runs (``GET /api/runs``)
  and returns one run with its event trail (``GET /api/runs/{run_id}``).
* ``run()`` defaults to loopback: this endpoint has no auth and spends money on LLM calls.

No test here performs a network call: the orchestrator is always a scripted stand-in and
the server is driven through ``TestClient``, which runs the ASGI app in-process.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from furrow.config import TaskModel, TaskStatus
from furrow.web import server

STOP_REASON = "goal complete: every planned task applied its edits and tests pass"

#: The exact event sequence ``Orchestrator`` emits for one successful cycle. Replaying it
#: keeps the streaming test honest about the wire format: every event frame must carry the
#: event name and the payload verbatim.
CYCLE_SCRIPT: tuple[tuple[str, dict[str, Any]], ...] = (
    ("cycle_start", {"cycle": 1, "focus": "ship it"}),
    ("plan", {"cycle": 1, "tasks": ["[t1] write the code"], "rationale": "one task"}),
    ("task_start", {"cycle": 1, "task": "t1"}),
    (
        "task_done",
        {
            "cycle": 1,
            "task": "t1",
            "status": "completed",
            "skipped": False,
            "result": "wrote a.py",
        },
    ),
    (
        "test",
        {"cycle": 1, "passed": True, "verified": True, "summary": "2 passed", "failures": []},
    ),
    ("cycle_end", {"cycle": 1, "changed_files": ["a.py"], "stalled_cycles": 0}),
    (
        "halt",
        {"reason": STOP_REASON, "cycles": 1, "changed_files": ["a.py"], "complete": True},
    ),
)


class ScriptedOrchestrator:
    """Replays :data:`CYCLE_SCRIPT` through ``on_event``, then reports completion."""

    instances: ClassVar[list[ScriptedOrchestrator]] = []

    def __init__(
        self,
        goal: str,
        client: object = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client
        self.on_event = on_event
        self.cycles = 0
        self.changed_files: list[str] = []
        self.tasks: list[TaskModel] = []
        self.stopped_reason: str | None = None
        ScriptedOrchestrator.instances.append(self)

    async def run(self) -> None:
        for name, payload in CYCLE_SCRIPT:
            self.cycles = int(payload.get("cycle", self.cycles))
            if self.on_event is not None:
                self.on_event(name, payload)
        self.changed_files = ["a.py"]
        self.tasks = [TaskModel(id="t1", description="write the code", status=TaskStatus.COMPLETED)]
        self.stopped_reason = STOP_REASON


class RaisingOrchestrator:
    """An engine that blows up inside ``run()``, the way infrastructure loss would."""

    instances: ClassVar[list[RaisingOrchestrator]] = []

    def __init__(self, goal: str, client: object = None, on_event: object = None) -> None:
        self.goal = goal
        self.on_event = on_event
        self.stopped_reason = "unexpected orchestrator error: RuntimeError: boom"
        RaisingOrchestrator.instances.append(self)

    async def run(self) -> None:
        raise RuntimeError("boom")


class HangingOrchestrator:
    """An engine that never finishes, so only a cancellation can end the run.

    The server drives it from another thread's event loop, so its lifecycle is reported
    through :class:`threading.Event` -- an ``asyncio.Event`` would belong to the server's
    loop, which this thread cannot await. :attr:`constructed` is the gate that says the
    engine exists at all; :meth:`await_instance` is the only supported way to reach one.
    """

    instances: ClassVar[list[HangingOrchestrator]] = []
    #: Opened by :meth:`__init__` once this engine is registered in :attr:`instances`.
    constructed: ClassVar[threading.Event] = threading.Event()

    def __init__(
        self,
        goal: str,
        client: object = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.goal = goal
        self.on_event = on_event
        self.started = threading.Event()
        self.cancelled = threading.Event()
        HangingOrchestrator.instances.append(self)
        HangingOrchestrator.constructed.set()

    async def run(self) -> None:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    @classmethod
    def reset(cls) -> None:
        """Drop every engine and close the gate again, scoping both to one test."""
        cls.instances.clear()
        cls.constructed = threading.Event()

    @classmethod
    def await_instance(cls, timeout: float = 5.0) -> HangingOrchestrator:
        """Block until the server's loop has built an engine, then return it.

        ``websocket.send_json`` only queues the message into the test client's stream: the
        run task that constructs the engine runs later, on the server's loop, so reading
        :attr:`instances` straight afterwards races it. Waiting on :attr:`constructed`
        removes that race and still fails fast, with a message, if no run ever starts.
        """
        assert cls.constructed.wait(timeout), "the server never started the hanging run"
        return cls.instances[0]


@pytest.fixture(autouse=True)
def reset_run_registry() -> Iterator[None]:
    """Empty the module-level run registry around every test.

    The registry is process-global by design -- it is exactly what ``GET /api/runs``
    reports -- so without this a run started by one test would appear in another's listing.
    The scripted engines are class-level for the same reason; resetting them here (and
    re-closing :attr:`HangingOrchestrator.constructed`) scopes them to the one test that
    uses them instead of relying on fixture ordering.
    """
    server._RUNS.clear()
    server._RUN_IDS = itertools.count(1)
    for stub in (ScriptedOrchestrator, RaisingOrchestrator):
        stub.instances.clear()
    HangingOrchestrator.reset()
    yield
    server._RUNS.clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A ``TestClient`` whose event loop outlives the single request it serves.

    ``with TestClient(app)`` is required rather than incidental: the portal (and its loop)
    is otherwise built per request, so the background task ``POST /api/start`` launches
    would be destroyed the instant its response was delivered.
    """
    with TestClient(server.app) as test_client:
        yield test_client


def use(monkeypatch: pytest.MonkeyPatch, stub: type) -> None:
    """Make the server build ``stub`` instead of the real orchestrator."""
    monkeypatch.setattr(server, "Orchestrator", stub)


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll until ``predicate`` holds; the server runs in another thread's event loop."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def drain(websocket: Any, limit: int = 20) -> list[dict[str, Any]]:
    """Read frames up to and including the first terminal one."""
    frames: list[dict[str, Any]] = []
    for _ in range(limit):
        payload = websocket.receive_json()
        frames.append(payload)
        if payload.get("type") in {"result", "error"}:
            break
    return frames


def test_index_serves_the_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert '<pre id="out">' in response.text
    assert "<h1>Furrow</h1>" in response.text


def test_index_uses_a_protocol_relative_websocket_scheme(client: TestClient) -> None:
    body = client.get("/").text

    assert "ws://" not in body
    assert "location.protocol" in body
    assert "'wss:'" in body


def test_index_appends_streamed_text_without_building_markup(client: TestClient) -> None:
    body = client.get("/").text

    assert "textContent +=" in body
    assert "innerHTML" not in body


def test_websocket_streams_events_and_a_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ScriptedOrchestrator)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "ship the feature"})
        frames = drain(websocket)

    events = [payload for payload in frames if payload["type"] == "event"]
    assert [payload["event"] for payload in events] == [name for name, _ in CYCLE_SCRIPT]
    assert events[0]["data"] == {"cycle": 1, "focus": "ship it"}
    assert isinstance(events[0]["ts"], float)
    assert events[0]["ts"] > 0
    assert frames[-1]["type"] == "result"
    assert frames[-1]["data"]["stopped_reason"] == STOP_REASON
    assert frames[-1]["data"]["cycles"] == 1
    assert frames[-1]["data"]["changed_files"] == ["a.py"]
    assert frames[-1]["data"]["tasks"] == [{"id": "t1", "status": "completed", "result": None}]


def test_websocket_rejects_an_empty_goal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ScriptedOrchestrator)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": ""})
        payload = websocket.receive_json()
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()

    assert payload["type"] == "error"
    assert "non-empty" in payload["data"]["message"]
    assert ScriptedOrchestrator.instances == []


def test_websocket_rejects_malformed_json(client: TestClient) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("{not json at all")
        payload = websocket.receive_json()
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()

    assert payload["type"] == "error"
    assert "must be JSON" in payload["data"]["message"]


def test_websocket_survives_an_orchestrator_exception(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, RaisingOrchestrator)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "explode"})
        frames = drain(websocket)

    assert [payload["type"] for payload in frames] == ["error"]
    assert frames[0]["data"]["message"] == "RuntimeError: boom"


def test_websocket_disconnect_cancels_the_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, HangingOrchestrator)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "run forever"})
        run = HangingOrchestrator.await_instance()
        assert run.started.wait(5)
        websocket.close(1000)
        assert run.cancelled.wait(5)

    record = server._RUNS["run-1"]
    assert record.runner is not None
    assert wait_for(record.runner.done)
    assert record.stopped_reason == server.CANCELLED_REASON


def test_websocket_stop_command_cancels_the_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, HangingOrchestrator)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "run forever"})
        run = HangingOrchestrator.await_instance()
        assert run.started.wait(5)
        websocket.send_json({"type": server.STOP_COMMAND})
        assert run.cancelled.wait(5)

    assert server._RUNS["run-1"].stopped_reason == server.CANCELLED_REASON


def test_post_api_start_and_list_runs(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, ScriptedOrchestrator)

    response = client.post("/api/start", json={"goal": "run in the background"})

    assert response.status_code == 202
    run_id = response.json()["id"]
    listing = client.get("/api/runs")
    assert listing.status_code == 200
    assert [entry["id"] for entry in listing.json()] == [run_id]
    detail = client.get(f"/api/runs/{run_id}")
    assert detail.json()["goal"] == "run in the background"
    assert wait_for(lambda: detail.json()["finished_at"] is not None)
    assert [entry["event"] for entry in detail.json()["events"]] == [
        name for name, _ in CYCLE_SCRIPT
    ]
    assert detail.json()["stopped_reason"] == STOP_REASON


def test_post_api_start_rejects_an_empty_goal(client: TestClient) -> None:
    response = client.post("/api/start", json={"goal": ""})

    assert response.status_code == 422
    assert client.get("/api/runs").json() == []


def test_get_an_unknown_run_is_a_404(client: TestClient) -> None:
    assert client.get("/api/runs/run-404").status_code == 404


def test_a_model_override_does_not_mutate_the_shared_settings(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ScriptedOrchestrator)
    before = server.settings.model_for("worker")

    client.post("/api/start", json={"goal": "use another model", "model": "override-model"})

    assert wait_for(lambda: bool(ScriptedOrchestrator.instances))
    override = ScriptedOrchestrator.instances[0].client
    assert override.settings.model_for("planner") == "override-model"
    assert override.settings.model_for("worker") == "override-model"
    assert server.settings.model_for("worker") == before


def test_run_defaults_to_loopback() -> None:
    default = inspect.signature(server.run).parameters["host"].default

    assert default == "127.0.0.1"
