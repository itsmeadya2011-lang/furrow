"""Tests for the web API improvements."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.web.server import app
from furrow.config import RunResult, Settings, TaskModel, Usage
from furrow.core.state import RunState, StateStore, HistoryEntry
from furrow.core.events import EventType


class TestWebAPICosts:
    """Tests for the /api/costs endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_api_costs_no_state(self, client, tmp_path, monkeypatch):
        """Test costs endpoint when no state exists."""
        # Mock the state store to return None
        with patch("furrow.web.server._state_store") as mock_store:
            mock_store_instance = MagicMock()
            mock_store_instance.load.return_value = None
            mock_store.return_value = mock_store_instance

            response = client.get("/api/costs")
            assert response.status_code == 200
            data = response.json()
            assert data["total"]["cost_usd"] == 0.0
            assert data["total"]["calls"] == 0
            assert data["by_cycle"] == []

    def test_api_costs_with_state(self, client, tmp_path, monkeypatch):
        """Test costs endpoint with persisted state."""
        # Create a mock state with history
        usage = Usage(input_tokens=1000, output_tokens=500, calls=5, cost_usd=0.01)
        history = [
            HistoryEntry(cycle=1, usage=Usage(input_tokens=500, output_tokens=250, calls=2, cost_usd=0.005)),
            HistoryEntry(cycle=2, usage=Usage(input_tokens=500, output_tokens=250, calls=3, cost_usd=0.005)),
        ]
        state = RunState(usage=usage, history=history)

        with patch("furrow.web.server._state_store") as mock_store:
            mock_store_instance = MagicMock()
            mock_store_instance.load.return_value = state
            mock_store.return_value = mock_store_instance

            with patch("furrow.web.server._CURRENT", {"orchestrator": None, "result": None}):
                response = client.get("/api/costs")
                assert response.status_code == 200
                data = response.json()
                assert data["total"]["cost_usd"] == 0.01
                assert data["total"]["calls"] == 5
                assert len(data["by_cycle"]) == 2
                assert data["by_cycle"][0]["cycle"] == 1
                assert data["by_cycle"][0]["cost_usd"] == 0.005
                assert data["by_cycle"][1]["cycle"] == 2


class TestWebAPIState:
    """Tests for the /api/state endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_api_state_no_state(self, client, tmp_path):
        """Test state endpoint when no state exists."""
        with patch("furrow.web.server._state_store") as mock_store:
            mock_store_instance = MagicMock()
            mock_store_instance.load.return_value = None
            mock_store_instance.exists.return_value = False
            mock_store.return_value = mock_store_instance

            response = client.get("/api/state")
            assert response.status_code == 200
            data = response.json()
            assert data["state"] is None
            assert data["exists"] is False

    def test_api_state_with_state(self, client, tmp_path):
        """Test state endpoint with persisted state."""
        state = RunState(
            goal="Test goal",
            original_goal="Original goal",
            cycles=2,
            tasks=[TaskModel(id="1", description="Task 1", status="completed")],
        )

        with patch("furrow.web.server._state_store") as mock_store:
            mock_store_instance = MagicMock()
            mock_store_instance.load.return_value = state
            mock_store_instance.exists.return_value = True
            mock_store.return_value = mock_store_instance

            response = client.get("/api/state")
            assert response.status_code == 200
            data = response.json()
            assert data["state"] is not None
            assert data["state"]["goal"] == "Test goal"
            assert data["state"]["cycles"] == 2
            assert data["exists"] is True


class TestWebAPIStatus:
    """Tests for the /api/status endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_api_status_no_run(self, client):
        """Test status endpoint when no run is active."""
        with patch("furrow.web.server._CURRENT", {"orchestrator": None, "result": None, "running": False}):
            response = client.get("/api/status")
            assert response.status_code == 200
            data = response.json()
            assert data["running"] is False
            assert data["status"] is None
            assert data["result"] is None

    def test_api_status_with_run(self, client):
        """Test status endpoint with active run."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.status.return_value = {"goal": "Test", "cycles": 1}

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"passed": True, "cycles": 1}

        with patch("furrow.web.server._CURRENT", {
            "orchestrator": mock_orchestrator,
            "result": mock_result,
            "running": True
        }):
            response = client.get("/api/status")
            assert response.status_code == 200
            data = response.json()
            assert data["running"] is True
            assert data["status"]["goal"] == "Test"
            assert data["result"]["passed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])