from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from furrow.web.server import WebSocketConsole


class FakeSettings(Settings):
    model: str = "fake-model"
    max_cycles: int = 1
    state_file: Path = Path("/tmp/furrow_test_state.json")


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_max_cycles():
    settings = FakeSettings()
    client = MagicMock(spec=LLMClient)
    planner = MagicMock()
    planner.plan = AsyncMock(
        return_value=Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
    )
    tester = MagicMock()
    tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))

    with patch("furrow.core.orchestrator.PlannerAgent", return_value=planner):
        with patch("furrow.core.orchestrator.TesterAgent", return_value=tester):
            orchestrator = Orchestrator(goal="test", client=client, settings=settings)
            orchestrator._cycle = AsyncMock()

            async def _run():
                await orchestrator.run()

            asyncio.run(_run())

    assert orchestrator.cycles >= settings.max_cycles


def test_orchestrator_state_persistence():
    settings = FakeSettings()
    state_file = settings.state_file
    if state_file.exists():
        state_file.unlink()

    client = MagicMock(spec=LLMClient)
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
    orchestrator = Orchestrator(goal="persist test", client=client, settings=settings)

    orchestrator._last_plan_rationale = plan.rationale
    orchestrator._current_tasks = plan.tasks
    for t in orchestrator._current_tasks:
        t.status = "completed"
        t.result = "done"
    orchestrator._save_state()

    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["goal"] == "persist test"
    assert data["cycles"] == 0
    assert len(data["tasks"]) == 1
    assert data["tasks"][0]["status"] == "completed"

    # Cleanup
    state_file.unlink()


def test_orchestrator_load_state():
    settings = FakeSettings()
    state_file = settings.state_file
    state_file.write_text(json.dumps({"goal": "loaded goal", "cycles": 5}))

    client = MagicMock(spec=LLMClient)
    orchestrator = Orchestrator(goal="original goal", client=client, settings=settings)

    assert orchestrator.goal == "loaded goal"
    assert orchestrator.cycles == 5

    # Cleanup
    state_file.unlink()


def test_orchestrator_model_override_does_not_mutate_global():
    original_model = FakeSettings().model
    client = MagicMock(spec=LLMClient)
    orchestrator = Orchestrator(goal="test", client=client, model="override-model", settings=FakeSettings())

    assert orchestrator.settings.model == "override-model"
    assert FakeSettings().model == original_model


@pytest.mark.asyncio
async def test_llm_ollama_routing():
    settings = FakeSettings()
    settings.provider = "ollama"
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="ollama response"))]
    client._ollama = MagicMock()
    client._ollama.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.complete("hello", model="llama3")

    assert result == "ollama response"
    client._ollama.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_websocket_console_sends_json():
    mock_ws = MagicMock()
    console = WebSocketConsole(mock_ws)
    console.print("hello world")

    # Give the scheduled task a chance to run
    await asyncio.sleep(0)

    mock_ws.send_json.assert_called_once()
    call_args = mock_ws.send_json.call_args[0][0]
    assert call_args["type"] == "log"
    assert call_args["message"] == "hello world"
