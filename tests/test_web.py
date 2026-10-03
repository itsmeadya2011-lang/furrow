from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from furrow.config import RunResult
from furrow.web.server import app


class FakeOrchestrator:
    _hang = False
    instances: list[FakeOrchestrator] = []

    def __init__(self, *, goal: str, on_progress, **kwargs) -> None:
        self.goal = goal
        self.on_progress = on_progress
        self.kwargs = kwargs
        self.stop_called = False
        self.hang = FakeOrchestrator._hang
        FakeOrchestrator.instances.append(self)

    async def run(self) -> RunResult:
        if self.hang:
            await asyncio.Event().wait()
        if self.on_progress:
            await self.on_progress("progress 1")
            await self.on_progress("progress 2")
        return RunResult(goal=self.goal, cycles=1, stopped_reason="goal_complete")

    async def stop(self) -> None:
        self.stop_called = True


@pytest.fixture(autouse=True)
def _patch_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOrchestrator.instances.clear()
    FakeOrchestrator._hang = False
    monkeypatch.setattr("furrow.web.server.Orchestrator", FakeOrchestrator)
    yield
    FakeOrchestrator.instances.clear()
    FakeOrchestrator._hang = False


def test_index_contains_expected_elements() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert 'id="goal"' in body
    assert 'id="start"' in body
    assert "/ws" in body
    assert "new WebSocket" in body


def test_websocket_receives_progress_messages() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text(json.dumps({"goal": "test goal"}))
        assert websocket.receive_text() == "progress 1"
        assert websocket.receive_text() == "progress 2"
        summary = websocket.receive_text()
        assert summary == "--- stopped: goal_complete after 1 cycle ---"


def test_websocket_missing_goal() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text(json.dumps({}))
        message = websocket.receive_text()
        assert message == "Error: 'goal' is required and must be a non-empty string."
        websocket.send_text(json.dumps({"goal": "real"}))
        assert websocket.receive_text() == "progress 1"


def test_websocket_blank_goal() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text('{"goal": "   "}')
        message = websocket.receive_text()
        assert message == "Error: 'goal' is required and must be a non-empty string."
        websocket.send_text(json.dumps({"goal": "real"}))
        assert websocket.receive_text() == "progress 1"


def test_websocket_non_json_frame() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("not json")
        message = websocket.receive_text()
        assert message == "Error: payload is not valid JSON."
        websocket.send_text(json.dumps({"goal": "real"}))
        assert websocket.receive_text() == "progress 1"


def test_websocket_non_object_json() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("[1, 2, 3]")
        message = websocket.receive_text()
        assert message == "Error: payload must be a JSON object."
        websocket.send_text(json.dumps({"goal": "real"}))
        assert websocket.receive_text() == "progress 1"


def test_websocket_disconnect_calls_stop() -> None:
    FakeOrchestrator._hang = True
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text(json.dumps({"goal": "test"}))
        websocket.close()
    time.sleep(0.1)
    assert FakeOrchestrator.instances[-1].stop_called is True


def test_websocket_summary_includes_stopped_reason() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text(json.dumps({"goal": "test"}))
        assert websocket.receive_text() == "progress 1"
        assert websocket.receive_text() == "progress 2"
        summary = websocket.receive_text()
        assert summary == "--- stopped: goal_complete after 1 cycle ---"


def test_run_defaults_to_settings() -> None:
    calls: list[tuple[str | None, int | None]] = []
    import uvicorn

    original_run = uvicorn.run

    def fake_run(app, host=None, port=None, **kwargs):
        calls.append((host, port))
        # Prevent uvicorn from actually starting.
        raise SystemExit(0)

    import furrow.web.server as server_module
    server_module.uvicorn.run = fake_run
    try:
        server_module.run()
    except SystemExit:
        pass
    finally:
        server_module.uvicorn.run = original_run
    assert calls == [("127.0.0.1", 8000)]
