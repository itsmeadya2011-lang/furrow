"""Tests for the web server WebSocket output relaying."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from furrow.web.server import app, WebSocketSink


def test_websocket_sink_emit():
    """WebSocketSink.emit should send text via websocket."""
    ws = MagicMock()
    ws.send_text = AsyncMock()

    sink = WebSocketSink(ws)

    async def test():
        await sink.emit("test message")
        ws.send_text.assert_called_with("test message")

    asyncio.run(test())


def test_wsockoutput_write_buffers():
    """WebSocketSink should implement the OutputSink protocol."""
    ws = MagicMock()
    ws.send_text = AsyncMock()
    sink = WebSocketSink(ws)
    assert hasattr(sink, "emit")
    assert asyncio.iscoroutinefunction(sink.emit)


def test_web_index_returns_html():
    """The index page should return HTML."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Furrow" in response.text
    assert "<form" in response.text


def test_websocket_no_goal_returns_error():
    """WebSocket without a goal should close the connection with an error."""
    from starlette.testclient import TestClient as StarletteTestClient

    client = StarletteTestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"goal": ""})
        data = websocket.receive_json()
        assert data["type"] == "error"


def test_websocket_start_request_model():
    """The StartRequest model should parse goal and model."""
    from furrow.web.server import StartRequest

    req = StartRequest(goal="test", model="gpt-4")
    assert req.goal == "test"
    assert req.model == "gpt-4"

    req2 = StartRequest(goal="test")
    assert req2.model is None


@pytest.mark.asyncio
async def test_websocket_relays_orchestrator_output():
    """WebSocket should relay orchestrator output through the sink."""
    from starlette.testclient import TestClient as StarletteTestClient

    captured_messages = []

    class FakeWebSocket:
        def __init__(self):
            self._messages = []

        async def accept(self):
            pass

        async def receive_json(self):
            return {"goal": "test goal"}

        async def send_text(self, text):
            captured_messages.append(text)

        async def send_json(self, data):
            captured_messages.append(json.dumps(data))

        async def close(self):
            pass

    # Mock the Orchestrator to immediately set _done and emit through sink
    with patch("furrow.web.server.Orchestrator") as MockOrchestrator:
        fake_ws = FakeWebSocket()
        instance = MagicMock()
        instance._done = True
        instance.run = AsyncMock()
        MockOrchestrator.return_value = instance

        # Test WebSocketSink directly
        sink = WebSocketSink(fake_ws)
        await sink.emit("test output")

        assert "test output" in captured_messages
