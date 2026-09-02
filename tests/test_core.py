import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.agents.prompts import PLANNER_PROMPT, WORKER_PROMPT
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_plan_with_dependencies():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b", dependencies=["1"]),
        ],
        rationale="r",
    )
    assert p.tasks[1].dependencies == ["1"]


def test_test_result_failures():
    t = TestResult(passed=False, summary="x", failures=["a", "b"])
    assert t.failures == ["a", "b"]


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks >= 1


def test_worker_prompt_has_json_shape():
    assert "files" in WORKER_PROMPT
    assert "content" in WORKER_PROMPT


def test_planner_prompt_has_rationale():
    assert "rationale" in PLANNER_PROMPT


def test_orchestrator_done_with_completed_tasks():
    client = MagicMock()
    client.settings.max_parallel_tasks = 5
    orch = Orchestrator(goal="g", client=client)
    orch._all_tasks = [TaskModel(id="1", description="x", status="completed")]
    assert orch._is_done() is True


def test_orchestrator_done_with_failed_tasks():
    client = MagicMock()
    client.settings.max_parallel_tasks = 5
    orch = Orchestrator(goal="g", client=client)
    orch._all_tasks = [TaskModel(id="1", description="x", status="failed")]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles():
    client = MagicMock()
    client.settings.max_parallel_tasks = 5
    client.settings.max_cycles = 1
    orch = Orchestrator(goal="g", client=client)
    orch._cycle = AsyncMock(return_value=None)
    await orch.run()
    assert orch.cycles == 1