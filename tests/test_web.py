import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from furrow.web.server import StartRequest, WebSocketConsole, app


class TestStartRequest:
    def test_start_request_valid(self):
        req = StartRequest(goal="test goal")
        assert req.goal == "test goal"
        assert req.model is None
        assert req.max_cycles is None
        assert req.max_parallel is None
        assert req.provider is None

    def test_start_request_with_all_fields(self):
        req = StartRequest(
            goal="test goal",
            model="custom-model",
            max_cycles=10,
            max_parallel=3,
            provider="openai",
        )
        assert req.goal == "test goal"
        assert req.model == "custom-model"
        assert req.max_cycles == 10
        assert req.max_parallel == 3
        assert req.provider == "openai"

    def test_start_request_goal_required(self):
        with pytest.raises(ValidationError):
            StartRequest()

    def test_start_request_extra_fields_ignored(self):
        req = StartRequest(goal="test", extra_field="ignored")
        assert req.goal == "test"


class TestWebSocketConsole:
    @pytest.fixture
    def mock_websocket(self):
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        return ws

    def test_websocket_console_init(self, mock_websocket):
        console = WebSocketConsole(mock_websocket)
        assert console._websocket == mock_websocket
        assert console.record is True
        assert console.width == 120
        assert console.force_terminal is False

    @pytest.mark.asyncio
    async def test_websocket_console_print_calls_send(self, mock_websocket):
        console = WebSocketConsole(mock_websocket)
        console.print("test message")

        # _send_text schedules a task via asyncio.get_running_loop().create_task
        mock_websocket.send_text.assert_called()

    @pytest.mark.asyncio
    async def test_websocket_console_print_runtime_error_handled(self, mock_websocket):
        mock_websocket.send_text.side_effect = RuntimeError("closed")
        console = WebSocketConsole(mock_websocket)
        console.print("test message")
        # Should not raise even if send_text fails


class TestWebApp:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_index_route(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Furrow" in response.text
        assert "ws://" in response.text

    def test_websocket_endpoint_exists(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.close()

    def test_app_title(self):
        assert app.title == "Furrow"


class TestWebSocketEndpoint:
    @pytest.mark.asyncio
    async def test_websocket_receives_start_data(self):
        from furrow.web.server import websocket_endpoint

        mock_websocket = AsyncMock()
        mock_websocket.receive_json = AsyncMock(
            return_value={
                "goal": "test goal",
                "provider": "anthropic",
                "max_cycles": 5,
                "max_parallel": 3,
            }
        )
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()

        with patch("furrow.web.server.Orchestrator") as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            try:
                await websocket_endpoint(mock_websocket)
            except Exception:
                pass  # Task may be cancelled

            mock_websocket.accept.assert_called_once()
            mock_orchestrator_class.assert_called_once()
            call_kwargs = mock_orchestrator_class.call_args.kwargs
            assert call_kwargs["goal"] == "test goal"
            assert call_kwargs["settings"].provider.value == "anthropic"
            assert call_kwargs["settings"].max_cycles == 5
            assert call_kwargs["settings"].max_parallel_tasks == 3

    @pytest.mark.asyncio
    async def test_websocket_with_minimal_data(self):
        from furrow.web.server import websocket_endpoint

        mock_websocket = AsyncMock()
        mock_websocket.receive_json = AsyncMock(return_value={"goal": "minimal goal"})
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()

        with patch("furrow.web.server.Orchestrator") as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            try:
                await websocket_endpoint(mock_websocket)
            except Exception:
                pass

            call_kwargs = mock_orchestrator_class.call_args.kwargs
            assert call_kwargs["goal"] == "minimal goal"
            assert call_kwargs["settings"].max_cycles == 0  # default
            assert call_kwargs["settings"].max_parallel_tasks == 5  # default

    @pytest.mark.asyncio
    async def test_websocket_disconnect_handled(self):
        from furrow.web.server import websocket_endpoint

        mock_websocket = AsyncMock()
        mock_websocket.receive_json = AsyncMock(side_effect=Exception("disconnect"))
        mock_websocket.accept = AsyncMock()

        await websocket_endpoint(mock_websocket)
        # Should not raise

    @pytest.mark.asyncio
    async def test_websocket_orchestrator_error_handled(self):
        from furrow.web.server import websocket_endpoint

        mock_websocket = AsyncMock()
        mock_websocket.receive_json = AsyncMock(return_value={"goal": "test"})
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()

        with patch("furrow.web.server.Orchestrator") as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run = AsyncMock(side_effect=RuntimeError("orchestrator error"))
            mock_orchestrator_class.return_value = mock_orchestrator

            try:
                await websocket_endpoint(mock_websocket)
            except Exception:
                pass

            # Should send error message
            mock_websocket.send_text.assert_any_call("[red]Orchestrator error: orchestrator error[/red]\n")


class TestRunFunction:
    @patch("furrow.web.server.uvicorn.run")
    def test_run_calls_uvicorn(self, mock_uvicorn_run):
        from furrow.web.server import run

        run(host="127.0.0.1", port=9000)
        mock_uvicorn_run.assert_called_once_with(app, host="127.0.0.1", port=9000)

    @patch("furrow.web.server.uvicorn.run")
    def test_run_defaults(self, mock_uvicorn_run):
        from furrow.web.server import run

        run()
        mock_uvicorn_run.assert_called_once_with(app, host="0.0.0.0", port=8000)


class TestWebSocketConsoleStaticMethod:
    def test_send_text_creates_task(self):
        mock_websocket = AsyncMock()
        mock_loop = MagicMock()
        mock_loop.create_task = MagicMock()

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            WebSocketConsole._send_text(mock_websocket, "test message")

        mock_loop.create_task.assert_called_once()
        # The task should be sending the message
        call_args = mock_loop.create_task.call_args[0][0]
        import asyncio
        assert asyncio.iscoroutine(call_args)
        # Clean up the coroutine
        call_args.close()