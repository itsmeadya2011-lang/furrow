"""Tests for the web server — index route and WebSocket endpoint."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from fastapi.testclient import TestClient

from furrow.web.server import app

client = TestClient(app)


class TestIndexRoute:
    def test_index_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Furrow" in response.text

    def test_index_has_goal_input(self):
        response = client.get("/")
        assert "goal" in response.text
        assert "Start" in response.text

    def test_index_has_websocket_script(self):
        response = client.get("/")
        assert "WebSocket" in response.text
        assert "/ws" in response.text


class TestWebSocketEndpoint:
    def test_websocket_receives_messages(self):
        """Verify that the WebSocket endpoint forwards orchestrator log messages."""

        class FakeOrchestrator:
            def __init__(self, *args, message_queue=None, **kwargs):
                self._queue = message_queue

            async def run(self):
                if self._queue:
                    self._queue.put_nowait("Planning...")
                    self._queue.put_nowait("Executing tasks...")
                    self._queue.put_nowait("Tests passed")
                    self._queue.put_nowait("Goal complete")

        with patch("furrow.web.server.Orchestrator", FakeOrchestrator):
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"goal": "test goal"})

                msg1 = websocket.receive_text()
                msg2 = websocket.receive_text()
                msg3 = websocket.receive_text()
                msg4 = websocket.receive_text()

                assert "Planning" in msg1
                assert "Executing" in msg2
                assert "Tests passed" in msg3
                assert "Goal complete" in msg4

    def test_websocket_no_goal_defaults_to_empty(self):
        """WebSocket with empty goal should connect and complete without errors."""

        class QuickOrchestrator:
            def __init__(self, *args, message_queue=None, **kwargs):
                self._queue = message_queue

            async def run(self):
                if self._queue:
                    self._queue.put_nowait("Starting...")

        with patch("furrow.web.server.Orchestrator", QuickOrchestrator):
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"goal": ""})
                msg = websocket.receive_text()
                assert "Starting" in msg

    def test_websocket_closes_after_orchestrator_done(self):
        """After the orchestrator finishes, the WebSocket should close cleanly."""

        class DoneOrchestrator:
            def __init__(self, *args, message_queue=None, **kwargs):
                self._queue = message_queue

            async def run(self):
                if self._queue:
                    self._queue.put_nowait("All done")

        with patch("furrow.web.server.Orchestrator", DoneOrchestrator):
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"goal": "finish"})
                msg = websocket.receive_text()
                assert "All done" in msg
