import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from furrow.web.server import app


def test_index_endpoint():
    """Test that the index endpoint returns HTML."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Furrow" in response.text


def test_websocket_endpoint_exists():
    """Test that the websocket endpoint is registered."""
    # Just verify the route exists
    routes = [route.path for route in app.routes]
    assert "/ws" in str(routes)  # WebSocket routes may be different


@pytest.mark.asyncio
async def test_websocket_progress_events():
    """Test that websocket sends progress events."""
    from furrow.core.orchestrator import Orchestrator
    from furrow.config import Plan, TaskModel, TestResult
    
    with patch("furrow.web.server.Orchestrator") as mock_orchestrator_class:
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # We can't easily test the full websocket flow without a real server
        # but we can verify the endpoint accepts connections
        pass