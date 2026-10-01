"""Tests for furrow.web.server module."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from furrow.web.server import app


# --- GET / ---

def test_index_returns_html():
    """GET / returns an HTML page."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Furrow" in response.text


def test_index_has_goal_form():
    """Index page contains a goal input field and submit button."""
    client = TestClient(app)
    response = client.get("/")
    html = response.text
    assert "goal" in html
    assert "submit" in html.lower()
    assert "Furrow" in html


def test_index_has_websocket_script():
    """Index page includes WebSocket JavaScript."""
    client = TestClient(app)
    response = client.get("/")
    assert "/ws" in response.text
    assert "WebSocket" in response.text


# --- WebSocket /ws ---

def test_websocket_accepts_connection():
    """WebSocket endpoint accepts connections and processes messages."""
    with patch("furrow.web.server.Orchestrator") as MockOrch:
        mock_instance = MockOrch.return_value
        mock_instance.run = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"goal": "test goal"})
            # If we reach here without raising, the connection was accepted
            # and the orchestrator.run() mock completed without error.


def test_websocket_empty_goal():
    """WebSocket endpoint handles empty goal gracefully."""
    with patch("furrow.web.server.Orchestrator") as MockOrch:
        mock_instance = MockOrch.return_value
        mock_instance.run = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"goal": ""})
            # Should not raise


def test_websocket_restores_console():
    """WebSocket endpoint restores the original console after completion."""
    import furrow.core.orchestrator as orch_module

    original = orch_module.console
    with patch("furrow.web.server.Orchestrator") as MockOrch:
        mock_instance = MockOrch.return_value
        mock_instance.run = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"goal": "test"})

    # The original console should be restored after the handler exits
    assert orch_module.console is original


def test_websocket_console_restored_on_error():
    """Console is restored even when an exception occurs during run()."""
    import furrow.core.orchestrator as orch_module

    original = orch_module.console
    with patch("furrow.web.server.Orchestrator") as MockOrch:
        mock_instance = MockOrch.return_value
        mock_instance.run = AsyncMock(side_effect=RuntimeError("simulated error"))

        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"goal": "test"})

    assert orch_module.console is original


# --- WebSocketOutput class ---

async def test_websocket_output_buffers_and_sends():
    """WebSocketOutput buffers incomplete lines and sends completed lines."""
    from furrow.web.server import WebSocketOutput

    sent_messages = []

    class FakeWebSocket:
        async def send_text(self, text):
            sent_messages.append(text)

    ws_output = WebSocketOutput(websocket=FakeWebSocket())

    ws_output.write("line1\n")
    ws_output.write("incomplete")
    ws_output.write(" line2\n")

    # Let pending async tasks (scheduled via asyncio.create_task) run
    await asyncio.sleep(0)

    assert "line1" in sent_messages
    assert "incomplete line2" in sent_messages
