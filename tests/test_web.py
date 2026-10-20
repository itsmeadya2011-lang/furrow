"""Tests for the Furrow web server.

Nothing here may touch the network and no real run may start: the orchestrator
is always replaced by :class:`FakeOrchestrator` and the settings point at a
temporary directory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from furrow.config import RunResult, settings
from furrow.core.events import Event, EventType
from furrow.core.state import RunState, StateStore
from furrow.web import server


class FakeOrchestrator:
    """Stand-in for the real orchestrator: records kwargs, emits two events."""

    init_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        state_store: Any = None,
        event_bus: Any = None,
        resume: bool = False,
    ) -> None:
        FakeOrchestrator.init_kwargs = {
            "goal": goal,
            "client": client,
            "settings": settings,
            "state_store": state_store,
            "event_bus": event_bus,
            "resume": resume,
        }
        self.goal = goal
        self.event_bus = event_bus
        self._status: dict[str, Any] = {
            "goal": goal,
            "original_goal": goal,
            "cycles": 0,
            "stopped_reason": "",
            "tasks": [],
        }

    def status(self) -> dict[str, Any]:
        return self._status

    async def run(self) -> RunResult:
        if self.event_bus is not None:
            self.event_bus.emit(Event(type=EventType.CYCLE_STARTED, data={"cycle": 1}))
            self.event_bus.emit(Event(type=EventType.TASK_FINISHED, data={"task": "1"}))
        return RunResult(
            passed=True,
            cycles=1,
            tasks_completed=1,
            summary="done",
            stopped_reason="goal complete",
        )


@pytest.fixture(autouse=True)
def isolated_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Path]:
    """Point the server at a temporary workspace and reset the run registry."""
    run_settings = settings.model_copy(update={"workspace": tmp_path})
    monkeypatch.setattr(server, "settings", run_settings)
    server._reset_registry()
    yield tmp_path
    server._reset_registry()


@pytest.fixture
def fake_orchestrator(monkeypatch: pytest.MonkeyPatch) -> type[FakeOrchestrator]:
    """Replace the orchestrator in the web layer with the fake one."""
    FakeOrchestrator.init_kwargs = {}
    monkeypatch.setattr("furrow.web.server.Orchestrator", FakeOrchestrator)
    return FakeOrchestrator


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client bound to the Furrow web app."""
    with TestClient(server.app) as test_client:
        yield test_client


def test_index_returns_html(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Furrow" in response.text


def test_api_state_without_state(client: TestClient) -> None:
    response = client.get("/api/state")

    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is False
    assert body["state"] is None


def test_api_state_with_state(client: TestClient, tmp_path: Path) -> None:
    store = StateStore(tmp_path / ".furrow" / "state.json")
    store.save(RunState(goal="ship it", original_goal="ship it", cycles=1))

    response = client.get("/api/state")

    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is True
    assert body["state"]["original_goal"] == "ship it"


def test_api_start_returns_started(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    response = client.post("/api/start", json={"goal": "build it"})

    assert response.status_code == 200
    body = response.json()
    assert body["started"] is True
    assert body["goal"] == "build it"
    kwargs = fake_orchestrator.init_kwargs
    assert kwargs["goal"] == "build it"
    assert kwargs["resume"] is False
    assert kwargs["event_bus"] is not None


def test_api_start_rejects_a_second_run(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    server._CURRENT["running"] = True

    response = client.post("/api/start", json={"goal": "build it"})

    assert response.status_code == 409
    body = response.json()
    assert body["started"] is False
    assert body["error"] == "a run is already in progress"


def test_api_status(client: TestClient, fake_orchestrator: type[FakeOrchestrator]) -> None:
    client.post("/api/start", json={"goal": "build it"})

    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert "running" in body
    assert "status" in body
    assert "result" in body


def test_websocket_streams_events(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    messages: list[dict[str, Any]] = []

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "g"})
        for _ in range(20):
            try:
                message = ws.receive_json()
            except WebSocketDisconnect:
                break
            messages.append(message)
            if message.get("type") == "done":
                break

    events = [message for message in messages if message.get("type") != "done"]
    assert events, f"expected at least one event message, got {messages}"
    assert any(message.get("type") == "cycle_started" for message in events)
    assert any(message.get("type") == "task_finished" for message in events)

    done = messages[-1]
    assert done["type"] == "done"
    assert done["message"] == "done"
    assert done["data"]["passed"] is True


def test_websocket_requires_goal(
    client: TestClient, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    messages: list[dict[str, Any]] = []

    with client.websocket_connect("/ws") as ws:
        ws.send_json({})
        try:
            messages.append(ws.receive_json())
        except WebSocketDisconnect:
            messages.append({"type": "closed"})

    assert messages, "the server must answer instead of hanging"
    assert "error" in json.dumps(messages[-1]).lower()
