from fastapi.testclient import TestClient

from furrow.web.server import app

client = TestClient(app)


def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    assert "Furrow" in response.text


def test_websocket_accepts():
    with client.websocket_connect("/ws"):
        pass
