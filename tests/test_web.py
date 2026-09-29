"""Tests for the web server WebSocket endpoint."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from furrow.web.server import app


def test_websocket_streams_messages() -> None:
    """WebSocket receives messages from the orchestrator's output_callback."""

    async def fake_run(output_callback=None):
        if output_callback:
            output_callback("message 1")
            output_callback("message 2")

    with patch("furrow.web.server.Orchestrator") as MockOrchestrator:
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(side_effect=fake_run)
        MockOrchestrator.return_value = mock_instance

        with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_json({"goal": "test goal"})
            messages: list[str] = []
            while True:
                try:
                    msg = websocket.receive_text()
                    messages.append(msg)
                except WebSocketDisconnect:
                    break

    assert "message 1" in messages
    assert "message 2" in messages
    assert "[DONE]" in messages


def test_index_page_returns_html() -> None:
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Furrow" in response.text
    assert "<form" in response.text
