"""Web server tests: fake orchestration over a real socket, no network, no LLM.

Every test swaps :class:`Orchestrator` for a fake that emits known event lines,
so nothing here builds an LLM client, reads an API key or touches the network.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from furrow.config import settings as global_settings
from furrow.web import server

GOAL = "add a health check endpoint"
EVENT_LINES = (
    "Cycle 1: planning",
    "Cycle 1: plan ready with 1 task(s)",
    "Cycle 1: tests passed (All tests passed)",
)


class Fake:
    """A stand-in orchestrator recording what the server handed it.

    Instances append themselves to ``made`` and expose ``ran``/``ended`` events so
    a test can synchronise on the server instead of sleeping.
    """

    made: list[Fake] = []

    def __init__(
        self,
        goal: str,
        client: Any = None,
        on_event: Any = None,
        settings: Any = None,
    ) -> None:
        self.goal = goal
        self.client = client
        self.on_event = on_event
        self.settings = settings
        self.stop_reason = "goal complete"
        self.is_done = True
        self.cancelled = False
        self.ran = threading.Event()
        self.ended = threading.Event()
        self.lines = list(type(self).events)
        type(self).made.append(self)

    def emit(self, line: str) -> None:
        if self.on_event is not None:
            self.on_event(line)

    async def run(self) -> None:
        self.ran.set()
        try:
            for line in self.lines:
                self.emit(line)
                await asyncio.sleep(0)
            if type(self).error is not None:
                raise type(self).error
            if type(self).hang:
                await asyncio.Event().wait()  # cancelled by the server, never returns
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.ended.set()


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: tuple[str, ...] = EVENT_LINES,
    error: Exception | None = None,
    hang: bool = False,
) -> list[Fake]:
    """Replace the server's orchestrator with a deterministic fake."""
    Fake.made = []
    Fake.events = events
    Fake.error = error
    Fake.hang = hang
    monkeypatch.setattr(server, "Orchestrator", Fake)
    return Fake.made


def read_until_terminal(ws: Any) -> list[dict[str, str]]:
    """Collect frames up to and including the first terminal one."""
    frames: list[dict[str, str]] = []
    while True:
        frame = ws.receive_json()
        frames.append(frame)
        if frame["type"] in server.TERMINAL_TYPES:
            return frames


def start_run(ws: Any, goal: str = GOAL, max_cycles: int | None = None) -> None:
    """Send the start frame and wait until the run is really under way.

    Consuming the ``started`` and first ``event`` frames makes what follows
    deterministic: only an orchestrator that exists can have emitted them, so
    ``fakes[0]`` is never a race against the server's construction.
    """
    payload: dict[str, Any] = {"goal": goal}
    if max_cycles is not None:
        payload["max_cycles"] = max_cycles
    ws.send_json(payload)
    assert ws.receive_json()["type"] == "started"
    assert ws.receive_json()["type"] == "event"


def kinds(frames: list[dict[str, str]]) -> list[str]:
    return [frame["type"] for frame in frames]


@pytest.fixture
def client() -> TestClient:
    with TestClient(server.app) as test_client:
        yield test_client


# ------------------------------------------------------------------- the page


def test_index_serves_the_ui_with_a_goal_input_and_a_websocket(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert 'id="goal"' in body
    assert "/ws" in body
    assert 'id="max-cycles"' in body
    assert 'id="stop"' in body
    # The bundle parses the same two-key envelope the server writes.
    assert "frame.type" in body and "frame.message" in body
    # ... and branches on every frame type the server can send.
    for kind in (
        server.TYPE_STARTED,
        server.TYPE_EVENT,
        server.TYPE_DONE,
        server.TYPE_STOPPED,
        server.TYPE_ERROR,
    ):
        assert f"'{kind}'" in body
    assert "{type: 'stop'}" in body
    assert "{{" not in body  # every placeholder was rendered


def test_index_documents_the_cycle_budget(client: TestClient) -> None:
    body = client.get("/").text

    assert f'value="{server.DEFAULT_MAX_CYCLES}"' in body
    assert str(server.DEFAULT_MAX_CYCLES) in body
    assert str(server.MAX_CYCLES_LIMIT) in body


# ------------------------------------------------------------------- streaming


def test_events_are_streamed_in_order_then_a_terminal_frame(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = install_fake(monkeypatch)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        frames = read_until_terminal(ws)

    assert fakes and fakes[0].goal == GOAL
    assert kinds(frames) == ["started", "event", "event", "event", "done"]
    assert [frame["message"] for frame in frames[1:4]] == list(EVENT_LINES)
    assert frames[-1]["message"] == "goal complete"
    assert GOAL in frames[0]["message"]


def test_every_frame_is_exactly_the_documented_envelope(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake(monkeypatch)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        frames = read_until_terminal(ws)

    for frame in frames:
        assert isinstance(frame, dict)
        assert set(frame) == {"type", "message"}  # the bundled JS reads exactly these
        assert isinstance(frame["type"], str)
        assert isinstance(frame["message"], str)
        assert frame["type"] in {
            server.TYPE_STARTED,
            server.TYPE_EVENT,
            server.TYPE_DONE,
            server.TYPE_STOPPED,
            server.TYPE_ERROR,
        }


def test_multi_line_events_are_collapsed_to_one_line(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake(monkeypatch, events=("line one\nline two\twith tabs",))

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        frames = read_until_terminal(ws)

    event = next(frame for frame in frames if frame["type"] == "event")
    assert event["message"] == "line one line two with tabs"


def test_an_orchestrator_exception_is_reported_as_an_error_frame(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake(monkeypatch, events=("before the boom",), error=RuntimeError("planner exploded"))

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        frames = read_until_terminal(ws)

    assert kinds(frames) == ["started", "event", "error"]
    assert "RuntimeError: planner exploded" in frames[-1]["message"]


def test_a_broken_start_message_is_reported_instead_of_crashing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = install_fake(monkeypatch)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"max_cycles": 2})
        frames = read_until_terminal(ws)

    assert kinds(frames) == ["error"]
    assert "goal" in frames[-1]["message"]
    assert fakes == []  # nothing was run


def test_an_empty_goal_is_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "   "})
        frames = read_until_terminal(ws)

    assert kinds(frames) == ["error"]


# ------------------------------------------------------------------------ stop


def test_a_stop_message_ends_the_run_and_closes_the_socket(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = install_fake(monkeypatch, hang=True)

    with client.websocket_connect("/ws") as ws:
        start_run(ws)
        assert not fakes[0].ended.is_set()
        ws.send_json({"type": "stop"})
        frames = read_until_terminal(ws)
        assert fakes[0].ended.wait(2), "the orchestrator task was not cancelled"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()  # the server closed the socket

    # start_run() already consumed "started" and the first "event".
    assert kinds(frames) == ["event", "event", "stopped"]
    assert frames[-1]["message"] == "stopped on client request"
    assert fakes[0].cancelled is True


def test_unknown_inbound_frames_do_not_break_a_running_loop(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = install_fake(monkeypatch, hang=True)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        ws.send_json({"type": "ping"})
        ws.send_text("not json at all")
        ws.send_json({"type": "stop"})
        frames = read_until_terminal(ws)
        assert fakes[0].ended.wait(2)

    assert frames[-1]["type"] == "stopped"


def test_client_disconnect_cancels_the_run(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = install_fake(monkeypatch, hang=True)

    with client.websocket_connect("/ws") as ws:
        start_run(ws)
        assert fakes[0].ran.is_set()
        ws.close(1000)
        assert fakes[0].ended.wait(2), "disconnect left the orchestrator running"

    assert fakes[0].cancelled is True


async def test_disconnect_leaves_no_task_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw ASGI view: after a disconnect nothing of the run may still be scheduled."""
    fakes = install_fake(monkeypatch, hang=True)
    inbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    outbound: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return await inbound.get()

    async def send(message: dict[str, Any]) -> None:
        outbound.append(message)

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": "/ws",
        "raw_path": b"/ws",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 5000),
        "server": ("testserver", 80),
        "subprotocols": [],
    }
    app_task = asyncio.create_task(server.app(scope, receive, send))
    await inbound.put({"type": "websocket.connect"})
    await inbound.put({"type": "websocket.receive", "text": json.dumps({"goal": GOAL})})
    await _until(lambda: any(m["type"] == "websocket.send" and "event" in m["text"] for m in outbound))
    assert fakes and fakes[0].ran.is_set()

    live = asyncio.all_tasks() - {asyncio.current_task(), app_task}
    await inbound.put({"type": "websocket.disconnect", "code": 1000})
    await asyncio.wait_for(app_task, 2)

    assert fakes[0].cancelled is True
    assert not (asyncio.all_tasks() - {asyncio.current_task()} - live)
    assert outbound[-1]["type"] == "websocket.close"


async def _until(predicate: Any, spins: int = 500) -> bool:
    """Give the event loop a bounded number of turns to satisfy ``predicate``."""
    for _ in range(spins):
        if predicate():
            return True
        await asyncio.sleep(0)
    return False


# ---------------------------------------------------------------- cycle budget


def test_the_requested_cycle_budget_is_applied(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = install_fake(monkeypatch)
    before = global_settings.max_cycles

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL, "max_cycles": 7})
        frames = read_until_terminal(ws)

    assert fakes[0].settings.max_cycles == 7
    assert fakes[0].settings is not global_settings  # a per-session copy
    assert global_settings.max_cycles == before, "the global settings singleton was mutated"
    assert "max_cycles: 7" in frames[0]["message"]


def test_an_absent_budget_falls_back_to_the_server_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = install_fake(monkeypatch)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        read_until_terminal(ws)

    assert fakes[0].settings.max_cycles == server.DEFAULT_MAX_CYCLES
    assert server.DEFAULT_MAX_CYCLES > 0  # a web session is never endless


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(0, 1), (-5, 1), (10_000, server.MAX_CYCLES_LIMIT), (1, 1)],
)
def test_a_client_budget_is_clamped(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, requested: int, expected: int
) -> None:
    fakes = install_fake(monkeypatch)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL, "max_cycles": requested})
        read_until_terminal(ws)

    assert fakes[0].settings.max_cycles == expected


def test_a_configured_budget_is_used_when_the_client_asks_for_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = install_fake(monkeypatch)
    monkeypatch.setattr(global_settings, "max_cycles", 4)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": GOAL})
        read_until_terminal(ws)

    assert fakes[0].settings.max_cycles == 4


# ------------------------------------------------------------------- the module


def test_run_binds_loopback_by_default_and_never_starts_on_import(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import inspect

    assert inspect.signature(server.run).parameters["host"].default == "127.0.0.1"

    served: list[tuple[str, int]] = []
    monkeypatch.setattr(server.uvicorn, "run", lambda app, host, port: served.append((host, port)))
    server.run()
    server.run(host="0.0.0.0", port=9001)

    assert served == [("127.0.0.1", 8000), ("0.0.0.0", 9001)]
    assert "unauthenticated" in capsys.readouterr().err  # warns about the non-loopback bind


def test_message_building_collapses_and_truncates() -> None:
    assert server._message("event", "  a\nb  ") == {"type": "event", "message": "a b"}
    long_line = server._message("event", "x" * 1000)["message"]
    assert len(long_line) == 500 and long_line.endswith("...")