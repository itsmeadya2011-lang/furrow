from __future__ import annotations

import pytest

from furrow.web.server import app


def test_routes_exist():
    paths = {r.path for r in app.routes}
    assert "/" in paths and "/health" in paths and "/ws" in paths


def test_health_endpoint():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("httpx not available")
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}
        r = client.get("/")
        assert r.status_code == 200
        assert "Furrow" in r.text
