"""Tests for the furrow web server (furrow.web.server)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from furrow.web.server import app


class FakeOrchestrator:
    """Stand-in for furrow.core.orchestrator.Orchestrator (never calls an LLM)."""

    goals: ClassVar[list[str]] = []

    def __init__(
        self,
        goal: str,
        on_event: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.on_event = on_event
        FakeOrchestrator.goals.append(goal)

    async def run(self) -> None:
        if self.on_event is not None:
            await self.on_event("hello event")


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake_orchestrator(monkeypatch) -> None:
    FakeOrchestrator.goals = []
    monkeypatch.setattr("furrow.web.server.Orchestrator", FakeOrchestrator)


def test_index_returns_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Furrow" in response.text


def test_websocket_run_streams_events(client: TestClient, fake_orchestrator) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"goal": "do a thing"})
        messages = [ws.receive_text() for _ in range(4)]

    assert messages == ["connected", "goal: do a thing", "hello event", "run finished"]
    assert FakeOrchestrator.goals == ["do a thing"]


def test_websocket_missing_goal_closes_with_error(
    client: TestClient,
    fake_orchestrator,
) -> None:
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_text() == "connected"
        ws.send_json({"not_a_goal": "x"})
        message = ws.receive_text()
        assert "error" in message
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()

    assert FakeOrchestrator.goals == []


def test_websocket_blank_goal_closes_with_error(
    client: TestClient,
    fake_orchestrator,
) -> None:
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_text() == "connected"
        ws.send_json({"goal": "   "})
        message = ws.receive_text()
        assert "error" in message
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()

    assert FakeOrchestrator.goals == []
