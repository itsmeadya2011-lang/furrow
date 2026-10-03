from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from furrow.web import server as server_module

# Hard cap on every blocking TestClient interaction: a regression in the /ws
# handler must fail the suite quickly instead of hanging it forever.
TIMEOUT = 10.0


class StubOrchestrator:
    """Stand-in for the real Orchestrator: records its inputs, emits two events."""

    instances: list[StubOrchestrator] = []

    def __init__(
        self,
        goal: str,
        settings: Any = None,
        on_event: Any = None,
    ) -> None:
        self.goal = goal
        self.settings = settings
        self.on_event = on_event
        self.ran = False
        self.cancelled = False
        StubOrchestrator.instances.append(self)

    async def run(self) -> str:
        self.ran = True
        try:
            if self.on_event is not None:
                await self.on_event({"type": "cycle_start", "cycle": 1, "goal": self.goal})
                await self.on_event({"type": "done", "reason": "stub"})
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "done"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    StubOrchestrator.instances.clear()
    monkeypatch.setattr(server_module, "Orchestrator", StubOrchestrator)
    return TestClient(server_module.app)


async def _call(fn, *args):
    """Run a blocking TestClient call under a hard timeout."""
    return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=TIMEOUT)


async def test_ws_streams_events_and_returns_promptly(client: TestClient) -> None:
    received: list[dict] = []

    def exercise() -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"goal": "ship it"})
            for _ in range(2):
                text = ws.receive_text()
                assert text.endswith("\n"), "events must be newline terminated"
                received.append(json.loads(text))

    await _call(exercise)

    assert [event["type"] for event in received] == ["cycle_start", "done"]
    assert received[0]["goal"] == "ship it"

    assert len(StubOrchestrator.instances) == 1
    orchestrator = StubOrchestrator.instances[0]
    assert orchestrator.goal == "ship it"
    assert orchestrator.on_event is not None
    assert orchestrator.settings is not None
    assert orchestrator.ran is True


async def test_ws_rejects_missing_goal_without_hanging(client: TestClient) -> None:
    received: list[dict] = []
    closed = False

    def exercise() -> None:
        nonlocal closed
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"model": "some-model"})
            received.append(json.loads(ws.receive_text()))
            try:
                ws.receive_text()
            except Exception:
                closed = True

    await _call(exercise)

    assert len(received) == 1
    assert received[0]["type"] == "error"
    assert received[0]["message"]
    assert closed is True
    assert StubOrchestrator.instances == []


async def test_ws_rejects_blank_goal(client: TestClient) -> None:
    received: list[dict] = []

    def exercise() -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"goal": "   "})
            received.append(json.loads(ws.receive_text()))

    await _call(exercise)

    assert received[0]["type"] == "error"
    assert StubOrchestrator.instances == []


async def test_start_endpoint_starts_background_run(client: TestClient) -> None:
    def exercise() -> Any:
        return client.post("/start", json={"goal": "x"})

    response = await _call(exercise)

    assert response.status_code == 200
    assert response.json() == {"status": "started", "goal": "x"}


async def test_start_endpoint_rejects_blank_goal(client: TestClient) -> None:
    def exercise() -> Any:
        return client.post("/start", json={"goal": "  "})

    response = await _call(exercise)

    assert response.status_code == 400


async def test_index_page_renders(client: TestClient) -> None:
    def exercise() -> Any:
        return client.get("/")

    response = await _call(exercise)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<h1>Furrow</h1>" in response.text


def test_model_override_copies_settings() -> None:
    base: str | None = getattr(server_module.settings, "model", None)

    copied = server_module.settings_for("override-model")
    same = server_module.settings_for(None)

    assert copied is not same
    assert copied.model == "override-model"
    assert server_module.settings_for(None) is server_module.settings
    if base is not None:
        assert server_module.settings.model == base