"""Tests for the web server."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.web.server import app
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent


def make_settings(**overrides):
    defaults = {
        "provider": Provider.ANTHROPIC,
        "model": "test-model",
        "anthropic_api_key": "test-key",
        "max_parallel_tasks": 5,
        "max_cycles": 0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_index_returns_html():
    """The index page returns HTML with a form."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Furrow" in response.text
    assert "form" in response.text.lower()


def test_start_request_model():
    """StartRequest model accepts goal and optional model."""
    from furrow.web.server import StartRequest
    req = StartRequest(goal="test goal", model="gpt-4")
    assert req.goal == "test goal"
    assert req.model == "gpt-4"

    req2 = StartRequest(goal="test goal")
    assert req2.model is None


@pytest.mark.asyncio
async def test_orchestrator_output_callback_with_web_style():
    """Test that Orchestrator with output_callback works in a web-like context.

    This indirectly tests the web server's use of the callback mechanism.
    """
    from furrow.core.orchestrator import Orchestrator

    settings = make_settings(max_cycles=1)
    received = []

    def callback(message: str) -> None:
        received.append(message)

    orch = Orchestrator(goal="test task", settings=settings, output_callback=callback)

    plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="test")
    pass_result = TestResult(passed=True, summary="ok")

    with patch.object(PlannerAgent, "plan", new=AsyncMock(return_value=plan)), \
         patch.object(WorkerAgent, "run", new=AsyncMock(return_value="done")), \
         patch.object(TesterAgent, "run", new=AsyncMock(return_value=pass_result)):
        await orch.run()

    assert len(received) > 0
    # Verify output contains cycle information
    combined = "\n".join(received)
    assert "Cycle" in combined or "Furrow" in combined
