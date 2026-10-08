from fastapi.testclient import TestClient

from furrow.web.server import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_html():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Furrow" in response.text
    assert "WebSocket" in response.text