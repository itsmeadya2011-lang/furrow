import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
    get_settings,
)
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = get_settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.workspace == Path.cwd()


def test_planner_agent_parses_json():
    client = MagicMock(spec=LLMClient)
    client.settings = Settings(provider="anthropic")
    client.complete = AsyncMock(
        return_value=json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "test", "files": [], "dependencies": []}
                ],
                "rationale": "ok",
            }
        )
    )

    agent = PlannerAgent(client=client)
    plan = asyncio.run(agent.plan("do something"))
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "test"
    client.complete.assert_called_once()


def test_planner_agent_raises_on_bad_json():
    client = MagicMock(spec=LLMClient)
    client.settings = Settings(provider="anthropic")
    client.complete = AsyncMock(return_value="not json")

    agent = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        asyncio.run(agent.plan("do something"))


def test_worker_agent_returns_result():
    client = MagicMock(spec=LLMClient)
    client.settings = Settings(provider="anthropic")
    client.complete = AsyncMock(return_value="did the thing")

    task = TaskModel(id="1", description="test task", files=["foo.py"])
    agent = WorkerAgent(task=task, client=client)
    result = asyncio.run(agent.run())
    assert result == "did the thing"


def test_worker_agent_writes_files(tmp_path: Path):
    client = MagicMock(spec=LLMClient)
    client.settings = Settings(provider="anthropic", workspace=tmp_path)
    client.complete = AsyncMock(
        return_value="--- src/main.py ---\nprint('hello')\n--- end ---\n\nSummary: added hello"
    )

    task = TaskModel(id="1", description="test task", files=["src/main.py"])
    agent = WorkerAgent(task=task, client=client)
    result = asyncio.run(agent.run())

    assert "Wrote src/main.py" in result
    assert "Summary: added hello" in result
    assert (tmp_path / "src" / "main.py").exists()
    assert (tmp_path / "src" / "main.py").read_text() == "print('hello')"


def test_tester_agent_passes():
    client = MagicMock(spec=LLMClient)
    client.settings = Settings(provider="anthropic")
    client.complete = AsyncMock(
        return_value=json.dumps(
            {"passed": True, "summary": "ok", "failures": []}
        )
    )

    agent = TesterAgent(client=client)
    with patch(
        "asyncio.create_subprocess_exec", new_callable=AsyncMock
    ) as mock_subprocess:
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_subprocess.return_value = mock_proc
        result = asyncio.run(agent.run("goal", []))

    assert result.passed is True
    assert result.summary == "ok"


def test_orchestrator_done_when_no_tasks():
    client = MagicMock(spec=LLMClient)
    client.settings = Settings(provider="anthropic")

    with patch.object(
        PlannerAgent, "plan", new_callable=AsyncMock
    ) as mock_plan:
        mock_plan.return_value = Plan(tasks=[], rationale="done")
        orch = Orchestrator(goal="test", client=client)
        asyncio.run(orch._cycle())
        assert orch._is_done()


def test_orchestrator_tracks_tasks():
    client = MagicMock(spec=LLMClient)
    client.settings = Settings(provider="anthropic")

    with patch.object(
        PlannerAgent, "plan", new_callable=AsyncMock
    ) as mock_plan:
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="t1", files=[]),
                TaskModel(id="2", description="t2", files=[]),
            ],
            rationale="ok",
        )
        mock_plan.return_value = plan

        with patch.object(
            WorkerAgent, "run", new_callable=AsyncMock
        ) as mock_worker:
            mock_worker.return_value = "done"

            with patch.object(
                TesterAgent, "run", new_callable=AsyncMock
            ) as mock_tester:
                mock_tester.return_value = TestResult(
                    passed=True, summary="ok", failures=[]
                )
                orch = Orchestrator(goal="test", client=client)
                asyncio.run(orch._cycle())
                assert orch._is_done()


def test_llm_client_raises_without_key():
    s = Settings(provider="anthropic", anthropic_api_key=None)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic
