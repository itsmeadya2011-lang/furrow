"""Tests for the web server FastAPI app."""


import pytest
from fastapi.testclient import TestClient

from furrow.web.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestWebServer:
    def test_index_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Furrow" in response.text
        assert "form" in response.text.lower()
        assert "goal" in response.text.lower()

    def test_websocket_endpoint(self, client, monkeypatch):
        """WebSocket should accept connection and receive streaming messages."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("FURROW_MAX_RETRIES", "1")
        with client.websocket_connect("/ws") as websocket:
            # Send initial goal
            websocket.send_json({"goal": "test goal"})

            # Should receive at least the initial panel message
            received = websocket.receive_text()
            assert "Furrow" in received or "Goal" in received
