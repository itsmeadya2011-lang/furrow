"""Tests for the FastAPI web layer (no LLM is ever contacted)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

import furrow.web.server as server
from furrow.core.events import Event


class StubOrchestrator:
    """Stands in for :class:`furrow.core.orchestrator.Orchestrator`."""

    hold_open = False

    def __init__(
        self,
        goal: str,
        client: Any = None,
        on_event: Any = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client
        self.on_event = on_event
        self.cycles = 0
        self.tasks: list[Any] = []
        self.history: list[Any] = []

    async def run(self) -> None:
        self.cycles += 1
        if self.on_event is not None:
            self.on_event(Event(kind="cycle_start", data={"cycle": 1}))
            self.on_event(Event(kind="done", data={"cycles": 1, "reason": "goal complete"}))
        if type(self).hold_open:
            await asyncio.sleep(0.2)


@pytest.fixture(autouse=True)
def stub_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "Orchestrator", StubOrchestrator)
    StubOrchestrator.hold_open = False
    manager = server.manager
    manager._subscribers.clear()
    manager._orchestrator = None
    manager._task = None


def wait_for_idle() -> None:
    deadline = time.monotonic() + 5
    while server.manager.running and time.monotonic() < deadline:
        time.sleep(0.01)


def test_health_reports_ok() -> None:
    with TestClient(server.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "running" in payload


def test_index_serves_the_dashboard() -> None:
    with TestClient(server.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Furrow" in response.text


def test_state_is_empty_before_a_run() -> None:
    with TestClient(server.app) as client:
        payload = client.get("/api/state").json()

    assert payload["goal"] == ""
    assert payload["cycles"] == 0
    assert payload["tasks"] == []


def test_websocket_delivers_events() -> None:
    with TestClient(server.app) as client, client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": "build a thing"})

        message = websocket.receive_json()

        assert "kind" in message
        assert message["kind"] in {"cycle_start", "done"}


def test_start_endpoint_reports_a_conflict_while_a_run_is_active() -> None:
    StubOrchestrator.hold_open = True
    with TestClient(server.app) as client:
        first = client.post("/api/start", json={"goal": "build a thing"})
        assert first.status_code == 202
        assert first.json()["started"] is True

        second = client.post("/api/start", json={"goal": "build another thing"})
        assert second.status_code == 409

        wait_for_idle()
