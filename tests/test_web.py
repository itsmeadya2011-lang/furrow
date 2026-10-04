"""The web layer: routes, the websocket event pump and the origin allowlist.

``furrow.web.server.Orchestrator`` is replaced by a recording stand-in, so a run can be
driven deterministically inside the ASGI test client. Everything else -- the handshake,
the payload validation, the event pump, the disconnect watcher and the shutdown path --
is the real server. Before the pump was wired up, the browser received nothing at all
and sat on "Starting..." forever, which is what the first websocket test locks in.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from furrow import __version__
from furrow.config import Settings
from furrow.web import server


@dataclass
class RecordingOrchestrator:
    """Stand-in for the real orchestrator that emits a scripted event stream."""

    goal: str
    on_event: Any = None
    block: bool = False
    fail: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)

    def emit(self, kind: str, message: str, **fields: Any) -> None:
        event = {"type": kind, "message": message}
        event.update(fields)
        self.on_event(event)

    def cancel(self) -> None:
        self.cancelled.set()
        self.finished.set()

    async def run(self) -> None:
        self.emit("run_start", f"Starting Furrow: {self.goal}", goal=self.goal)
        if self.block:
            await self.finished.wait()
        if self.fail:
            raise RuntimeError("the provider is on fire")
        self.emit("cycle_start", "=== Cycle 1 ===", cycle=1)
        self.emit("run_done", "Goal complete after 1 cycle(s).", complete=True, cycles=1)


@pytest.fixture(autouse=True)
def no_run_carried_over() -> Iterator[None]:
    """Keep the one-run-at-a-time module state out of neighbouring tests."""
    server._release_run()
    yield
    server._release_run()


@pytest.fixture
def orchestrators(monkeypatch: pytest.MonkeyPatch) -> list[RecordingOrchestrator]:
    """Replace the orchestrator with a recording stand-in and collect the instances."""
    built: list[RecordingOrchestrator] = []

    def factory(**kwargs: Any) -> RecordingOrchestrator:
        instance = RecordingOrchestrator(**kwargs)
        built.append(instance)
        return instance

    monkeypatch.setattr(server, "Orchestrator", factory)
    return built


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(server.app) as test_client:
        yield test_client


# ------------------------------------------------------------------- routes


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Furrow</title>" in response.text


def test_index_page_upgrades_to_a_secure_socket_on_an_https_page(client: TestClient) -> None:
    html = client.get("/").text
    assert "wss://" in html
    assert "location.protocol === 'https:'" in html
    assert "'/ws'" in html


def test_health_reports_status_version_and_idleness(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "active_run": False}


def test_health_reports_an_active_run_without_leaking_its_goal(
    client: TestClient,
) -> None:
    assert server._claim_run("a secret objective") is True
    body = client.get("/health").json()
    assert body["active_run"] is True
    assert "a secret objective" not in client.get("/health").text


# ---------------------------------------------------------------- websocket


def test_the_client_receives_the_events_the_orchestrator_emits(
    client: TestClient, orchestrators: list[RecordingOrchestrator]
) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "add a widget"})
        received = [websocket.receive_json() for _ in range(3)]
        closed = websocket.receive()

    assert [event["type"] for event in received] == ["run_start", "cycle_start", "run_done"]
    assert received[0]["goal"] == "add a widget"
    assert received[-1]["complete"] is True
    assert closed["type"] == "websocket.close"
    assert orchestrators[0].goal == "add a widget"


def test_the_run_is_freed_for_the_next_client(
    client: TestClient, orchestrators: list[RecordingOrchestrator]
) -> None:
    for _ in range(2):
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"goal": "twice"})
            websocket.receive_json()
    assert len(orchestrators) == 2


def test_a_run_that_blows_up_reports_an_error_instead_of_a_traceback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server, "Orchestrator", lambda **kwargs: RecordingOrchestrator(**kwargs, fail=True)
    )
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "boom"})
        events = [websocket.receive_json() for _ in range(2)]
        closed = websocket.receive()

    assert events[-1]["type"] == "error"
    assert "the provider is on fire" in events[-1]["message"]
    assert closed["type"] == "websocket.close"


def test_a_second_run_is_refused_while_one_is_active(
    client: TestClient, orchestrators: list[RecordingOrchestrator]
) -> None:
    assert server._claim_run("already busy") is True
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "second"})
        error = websocket.receive_json()
        closed = websocket.receive()

    assert error["type"] == "error"
    assert "already in progress" in error["message"]
    assert closed["code"] == server.WS_CLOSE_DENIED
    assert orchestrators == []


def test_a_disconnect_cancels_the_orchestrator(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked: list[RecordingOrchestrator] = []

    def factory(**kwargs: Any) -> RecordingOrchestrator:
        instance = RecordingOrchestrator(**kwargs, block=True)
        blocked.append(instance)
        return instance

    monkeypatch.setattr(server, "Orchestrator", factory)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "long run"})
        assert websocket.receive_json()["type"] == "run_start"
    assert blocked[0].cancelled.wait(timeout=10) is True


def test_a_failed_handover_reports_the_reason_without_a_traceback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refusing(**kwargs: Any) -> RecordingOrchestrator:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(server, "Orchestrator", refusing)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "no credentials"})
        error = websocket.receive_json()
        closed = websocket.receive()

    assert error["type"] == "error"
    assert "ANTHROPIC_API_KEY is not set" in error["message"]
    assert closed["code"] == server.WS_CLOSE_DENIED


# ------------------------------------------------------- start-request errors


def test_malformed_json_is_refused_with_a_clean_error(
    client: TestClient, orchestrators: list[RecordingOrchestrator]
) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("{not json at all")
        error = websocket.receive_json()
        closed = websocket.receive()

    assert error == {"type": "error", "message": "Start request must be valid JSON."}
    assert closed["code"] == server.WS_CLOSE_INVALID_PAYLOAD
    assert "Traceback" not in closed["reason"]
    assert orchestrators == []


@pytest.mark.parametrize("payload", ["[1, 2]", '"a string"', "17", "null"])
def test_a_non_object_payload_is_refused(
    client: TestClient, payload: str, orchestrators: list[RecordingOrchestrator]
) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text(payload)
        error = websocket.receive_json()
        websocket.receive()

    assert error["message"] == "Start request must be a JSON object."
    assert orchestrators == []


def test_an_empty_goal_is_refused(
    client: TestClient, orchestrators: list[RecordingOrchestrator]
) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": ""})
        error = websocket.receive_json()
        closed = websocket.receive()

    assert error["message"].startswith("Invalid start request (goal:")
    assert closed["code"] == server.WS_CLOSE_INVALID_PAYLOAD
    assert orchestrators == []


@pytest.mark.parametrize("goal", ["   ", "\n\t "])
def test_a_whitespace_only_goal_is_refused(
    client: TestClient, goal: str, orchestrators: list[RecordingOrchestrator]
) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": goal})
        error = websocket.receive_json()
        closed = websocket.receive()

    assert error["message"] == "Invalid start request (goal: goal must not be blank)"
    assert closed["code"] == server.WS_CLOSE_INVALID_PAYLOAD
    assert orchestrators == []


def test_an_over_long_goal_is_refused(
    client: TestClient, orchestrators: list[RecordingOrchestrator]
) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "x" * (server.MAX_GOAL_CHARS + 1)})
        error = websocket.receive_json()
        websocket.receive()

    assert "Invalid start request" in error["message"]
    assert orchestrators == []


def test_a_missing_goal_field_is_refused(
    client: TestClient, orchestrators: list[RecordingOrchestrator]
) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"objective": "wrong field"})
        error = websocket.receive_json()
        websocket.receive()

    assert "Invalid start request" in error["message"]
    assert orchestrators == []


def test_a_goal_is_stripped_before_the_run_starts(
    client: TestClient, orchestrators: list[RecordingOrchestrator]
) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "  add a widget  "})
        websocket.receive_json()
    assert orchestrators[0].goal == "add a widget"


# -------------------------------------------------------------- origin checks


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://127.0.0.1",
        "http://LOCALHOST",
        "https://[::1]:8000",
    ],
)
def test_a_loopback_or_absent_origin_is_allowed(origin: str | None) -> None:
    assert server._origin_allowed(origin) is True


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        "https://furrow.attacker.test",
        "http://127.0.0.1.evil.example",
        "http://127.0.0.2",
        "ws://127.0.0.1:8000",
        "file:///etc/passwd",
        "null",
    ],
)
def test_a_foreign_origin_is_refused(origin: str) -> None:
    assert server._origin_allowed(origin) is False


def test_a_foreign_origin_is_refused_before_the_run_starts(
    client: TestClient, orchestrators: list[RecordingOrchestrator]
) -> None:
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
            ws.send_json({"goal": "cross-site"})
            ws.receive()
    assert excinfo.value.code == server.WS_CLOSE_DENIED
    assert orchestrators == []


def test_a_foreign_origin_is_logged_as_a_warning(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=server.__name__):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}):
                pass
    assert "rejected websocket handshake" in caplog.text


def test_a_loopback_origin_is_accepted(
    client: TestClient, orchestrators: list[RecordingOrchestrator]
) -> None:
    with client.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:8000"}) as ws:
        ws.send_json({"goal": "same origin"})
        assert ws.receive_json()["type"] == "run_start"
    assert len(orchestrators) == 1


# ------------------------------------------------------------------- run()


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Capture what ``run()`` would hand to uvicorn."""
    captured: list[tuple[str, int]] = []
    monkeypatch.setattr(
        server.uvicorn, "run", lambda app, host, port: captured.append((host, port))
    )
    return captured


def test_run_binds_loopback_by_default(
    served: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "get_settings", lambda: Settings())
    server.run()
    assert served == [("127.0.0.1", 8000)]


def test_run_honours_an_explicit_host_and_port(
    served: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "get_settings", lambda: Settings())
    server.run(host="localhost", port=8123)
    assert served == [("localhost", 8123)]


def test_run_takes_the_default_port_from_settings(
    served: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: Settings(web_port=9001, web_host="localhost"),
    )
    server.run()
    assert served == [("localhost", 9001)]


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_run_refuses_an_out_of_range_port(served: list[tuple[str, int]], port: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 65535"):
        server.run(port=port)


def test_run_warns_when_it_is_not_bound_to_loopback(
    served: list[tuple[str, int]], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=server.__name__):
        server.run(host="0.0.0.0", port=8000)
    assert "exposes an unauthenticated agent" in caplog.text


def test_run_does_not_warn_on_loopback(
    served: list[tuple[str, int]], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=server.__name__):
        server.run(host="127.0.0.1", port=8000)
    assert "unauthenticated agent" not in caplog.text
