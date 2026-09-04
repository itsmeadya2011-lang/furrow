from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_orchestrator_task_state(tmp_path: Path) -> None:
    fake_plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    fake_test = TestResult(passed=True, summary="all good", failures=[])

    mock_planner = AsyncMock()
    mock_planner.plan.return_value = fake_plan
    mock_tester = AsyncMock()
    mock_tester.run.return_value = fake_test
    mock_worker = AsyncMock(return_value="done")

    orchestrator = Orchestrator(goal="test goal", state_path=tmp_path / "state.json")
    orchestrator.planner = mock_planner
    with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            await orchestrator._cycle()

    assert len(orchestrator.tasks) == 1
    assert orchestrator.tasks[0].status == "completed"
    assert orchestrator.tasks[0].result == "done"
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_max_cycles(tmp_path: Path) -> None:
    fake_plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    fake_test = TestResult(passed=True, summary="ok", failures=[])

    mock_planner = AsyncMock()
    mock_planner.plan.return_value = fake_plan
    mock_tester = AsyncMock()
    mock_tester.run.return_value = fake_test
    # Worker fails so task is marked failed, preventing _is_done() from returning True
    mock_worker = AsyncMock(side_effect=Exception("worker failed"))

    settings = Settings(max_cycles=1)
    orchestrator = Orchestrator(goal="test goal", client=LLMClient(settings=settings), state_path=tmp_path / "state.json")
    orchestrator.planner = mock_planner
    with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            await orchestrator.run()

    assert orchestrator.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_state_persistence(tmp_path: Path) -> None:
    fake_plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    fake_test = TestResult(passed=True, summary="ok", failures=[])

    mock_planner = AsyncMock()
    mock_planner.plan.return_value = fake_plan
    mock_tester = AsyncMock()
    mock_tester.run.return_value = fake_test
    mock_worker = AsyncMock(return_value="done")

    state_path = tmp_path / "state.json"
    orchestrator = Orchestrator(goal="test goal", state_path=state_path)
    orchestrator.planner = mock_planner
    with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            await orchestrator._cycle()

    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert data["goal"] == "test goal"
    assert len(data["tasks"]) == 1
    assert data["tasks"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_llm_ollama_complete() -> None:
    settings = Settings(provider="ollama", ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)

    mock_response = type("Resp", (), {})()
    mock_response.json = lambda: {"response": "hello from ollama"}
    mock_response.raise_for_status = lambda: None

    mock_post = AsyncMock(return_value=mock_response)
    mock_client = AsyncMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("furrow.llm.httpx.AsyncClient", return_value=mock_client):
        result = await client.complete("say hi")

    assert result == "hello from ollama"
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["json"]["model"] == settings.model


@pytest.mark.asyncio
async def test_tester_detects_lockfiles(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{}")
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'test'")

    settings = Settings(workspace=tmp_path)
    agent = TesterAgent(settings=settings)
    commands = agent._detect_test_commands(tmp_path)

    assert ["npm", "test", "--", "--silent"] in commands
    assert ["pnpm", "test", "--", "--silent"] in commands
    assert ["yarn", "test", "--silent"] in commands
    assert ["cargo", "test", "-q"] in commands
    assert ["pytest", "-q"] in commands


@pytest.mark.asyncio
async def test_tester_no_test_runner(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path)
    agent = TesterAgent(settings=settings)
    result = await agent._run_tests()
    assert result == "No test runner found."
