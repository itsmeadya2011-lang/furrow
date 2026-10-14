from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from furrow.config import Settings
from furrow.web import StartRequest, _build_settings, app


class TestWeb:
    def test_health_endpoint(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_websocket_streaming_receives_events(self):
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send a StartRequest to trigger the orchestrator
            websocket.send_json({"goal": "Test goal"})
            data = websocket.receive_json()
            assert "event" in data

    def test_start_request_model_all_fields(self):
        request = StartRequest(
            goal="Test goal",
            provider="openai",
            model="gpt-4",
            max_cycles=5,
            max_parallel_tasks=3,
            test_command="pytest -v",
            test_timeout=60,
            workspace="/tmp/workspace",
        )
        assert request.goal == "Test goal"
        assert request.provider == "openai"
        assert request.model == "gpt-4"
        assert request.max_cycles == 5
        assert request.max_parallel_tasks == 3
        assert request.test_command == "pytest -v"
        assert request.test_timeout == 60
        assert request.workspace == "/tmp/workspace"

    def test_start_request_model_defaults(self):
        request = StartRequest(goal="Test goal")
        assert request.goal == "Test goal"
        assert request.provider is None
        assert request.model is None
        assert request.max_cycles is None
        assert request.max_parallel_tasks is None
        assert request.test_command is None
        assert request.test_timeout is None
        assert request.workspace is None

    def test_build_settings_applies_overrides(self):
        base = Settings(anthropic_api_key="base-key")
        request = StartRequest(
            goal="Test",
            provider="openai",
            model="gpt-4",
            max_cycles=10,
            max_parallel_tasks=8,
            test_command="custom-test",
            test_timeout=300,
            workspace="/custom/path",
        )

        settings = _build_settings(base, request)

        assert settings.provider.value == "openai"
        assert settings.model == "gpt-4"
        assert settings.max_cycles == 10
        assert settings.max_parallel_tasks == 8
        assert settings.test_command == "custom-test"
        assert settings.test_timeout == 300
        assert str(settings.workspace) == "/custom/path"

    def test_build_settings_keeps_base_when_override_none(self):
        base = Settings(provider="anthropic", model="claude-3", max_cycles=3, max_parallel_tasks=2)
        request = StartRequest(goal="Test")

        settings = _build_settings(base, request)

        assert settings.provider.value == "anthropic"
        assert settings.model == "claude-3"
        assert settings.max_cycles == 3
        assert settings.max_parallel_tasks == 2

    def test_build_settings_provider_string_to_enum(self):
        base = Settings()
        request = StartRequest(goal="Test", provider="ollama")

        settings = _build_settings(base, request)

        assert settings.provider.value == "ollama"

    @pytest.mark.asyncio
    async def test_start_endpoint_creates_orchestrator(self):
        from furrow.web import start_orchestrator

        request = StartRequest(goal="Test goal")

        with patch("furrow.web.server.Orchestrator") as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            await start_orchestrator(request)

            mock_orchestrator_class.assert_called_once()
            mock_orchestrator.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_endpoint_passes_callback(self):
        from furrow.web import start_orchestrator

        request = StartRequest(goal="Test goal")

        with patch("furrow.web.server.Orchestrator") as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            await start_orchestrator(request)

            # Verify orchestrator was created with goal and client
            mock_orchestrator_class.assert_called_once()
            call_kwargs = mock_orchestrator_class.call_args[1]
            assert call_kwargs["goal"] == "Test goal"
            assert "client" in call_kwargs

    def test_start_endpoint_validation_error(self):
        client = TestClient(app)
        response = client.post("/start", json={})
        assert response.status_code == 422  # Validation error

    def test_start_endpoint_returns_task_id(self):
        client = TestClient(app)

        with patch("furrow.core.orchestrator.Orchestrator") as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            response = client.post("/start", json={"goal": "Test goal"})
            assert response.status_code == 200
            data = response.json()
            assert "task_id" in data
            assert isinstance(data["task_id"], str)