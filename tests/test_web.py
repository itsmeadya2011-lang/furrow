"""Tests for the web server and output callback."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import TaskModel
from furrow.core.orchestrator import Orchestrator
from furrow.web.server import app


class TestOrchestratorOutputCallback:
    @pytest.mark.asyncio
    async def test_output_callback_receives_strings(self):
        """String messages are forwarded to the callback."""
        received: list[str] = []

        def callback(msg: str):
            received.append(msg)

        plan = MagicMock()
        plan.tasks = []
        plan.model_dump.return_value = {"tasks": [], "rationale": "done"}

        orch = Orchestrator(goal="test", output_callback=callback)
        with patch.object(orch.planner, "plan", new_callable=AsyncMock, return_value=plan):
            await orch._cycle()

        # Should have received at least the cycle output
        assert len(received) > 0

    @pytest.mark.asyncio
    async def test_output_callback_receives_rich_objects_as_text(self):
        """Rich objects (Panel, etc.) are converted to plain text for the callback."""
        received: list[str] = []

        def callback(msg: str):
            received.append(msg)

        orch = Orchestrator(goal="test", output_callback=callback)
        orch._output("Hello [bold]world[/bold]")

        # Should have received the plain text version
        assert len(received) == 1
        assert "Hello" in received[0]

    @pytest.mark.asyncio
    async def test_output_callback_async(self):
        """Async callbacks are scheduled as tasks."""
        received: list[str] = []

        async def callback(msg: str):
            received.append(msg)

        orch = Orchestrator(goal="test", output_callback=callback)
        orch._output("test message")

        # The async callback is scheduled as a task
        # Give it a moment to complete
        import asyncio
        await asyncio.sleep(0.1)
        assert len(received) == 1
        assert "test message" in received[0]

    @pytest.mark.asyncio
    async def test_no_callback_works(self):
        """Orchestrator works fine without a callback."""
        plan = MagicMock()
        plan.tasks = []
        plan.model_dump.return_value = {"tasks": [], "rationale": "done"}

        orch = Orchestrator(goal="test")
        with patch.object(orch.planner, "plan", new_callable=AsyncMock, return_value=plan):
            await orch._cycle()

        assert orch._is_done()


class TestWebServer:
    def test_app_exists(self):
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)

    def test_start_request_model(self):
        from furrow.web.server import StartRequest
        req = StartRequest(goal="Add auth")
        assert req.goal == "Add auth"
        assert req.model is None

    def test_websocket_route_exists(self):
        """Verify the WebSocket route is registered."""
        routes = [r.path for r in app.routes]
        assert "/ws" in routes
        assert "/" in routes
