from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from furrow.config import Provider, Settings
from furrow.web.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_index_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Furrow" in response.text
    assert "goal" in response.text.lower()


def test_websocket_streaming(client: TestClient) -> None:
    captured: dict[str, object] = {}

    def fake_orchestrator(goal: str, output_callback=None, **kwargs):
        captured["goal"] = goal
        captured["callback"] = output_callback
        fake = MagicMock()

        async def run() -> None:
            if output_callback:
                await output_callback("line-1")
                await output_callback("line-2")

        fake.run = run
        return fake

    with patch("furrow.web.server.Orchestrator", side_effect=fake_orchestrator):
        with client.websocket_connect("/ws") as session:
            session.send_json({"goal": "build a widget"})
            messages = [session.receive_text(), session.receive_text()]

    assert captured["goal"] == "build a widget"
    assert messages == ["line-1", "line-2"]


def test_websocket_model_override(client: TestClient) -> None:
    real_settings = Settings(
        provider=Provider.ANTHROPIC,
        model="default-model",
    )

    captured: dict[str, object] = {}

    def fake_orchestrator(goal: str, output_callback=None, **kwargs):
        captured["goal"] = goal
        captured["callback"] = output_callback
        fake = MagicMock()

        async def run() -> None:
            # The real run() would block; here we just return immediately so the
            # websocket endpoint closes cleanly and the test does not hang.
            pass

        fake.run = run
        return fake

    with patch("furrow.web.server.settings", real_settings):
        with patch("furrow.web.server.Orchestrator", side_effect=fake_orchestrator):
            with client.websocket_connect("/ws") as session:
                session.send_json({"goal": "do thing", "model": "override-model"})

    assert real_settings.model == "override-model"
    assert captured["goal"] == "do thing"