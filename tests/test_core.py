import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from furrow.config import Plan, TaskModel, TestResult, Settings
from furrow.llm import strip_code_fences
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_strip_code_fences():
    # Strips a ```json fenced block
    json_fenced = '```json\n{"key": "value"}\n```'
    assert strip_code_fences(json_fenced) == '{"key": "value"}'

    # Strips a plain ``` fence
    plain_fenced = '```\nplain content\n```'
    assert strip_code_fences(plain_fenced) == "plain content"

    # Strips surrounding whitespace
    with_whitespace = '  \n  ```json\n{"key": "value"}\n```  \n  '
    assert strip_code_fences(with_whitespace) == '{"key": "value"}'

    # Leaves plain JSON unchanged
    plain_json = '{"key": "value"}'
    assert strip_code_fences(plain_json) == plain_json


def test_strip_code_fences_noop():
    # Plain text without fences is returned trimmed
    plain_text = "  some text without fences  "
    assert strip_code_fences(plain_text) == "some text without fences"


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.request_timeout == 300
    assert s.max_retries == 3
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


async def test_planner_validation_unique_ids():
    # Stub LLMClient that returns duplicate task IDs
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(
        return_value='{"tasks": [{"id": "1", "description": "a"}, {"id": "1", "description": "b"}], "rationale": "r"}'
    )
    mock_client.settings = AsyncMock()
    mock_client.settings.planner_model = "test-model"

    planner = PlannerAgent(client=mock_client)

    with pytest.raises(ValueError, match="duplicate task ids"):
        await planner.plan("some goal")


async def test_planner_rejects_empty_tasks():
    # Stub LLMClient that returns empty tasks list
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(
        return_value='{"tasks": [], "rationale": "done"}'
    )
    mock_client.settings = AsyncMock()
    mock_client.settings.planner_model = "test-model"

    planner = PlannerAgent(client=mock_client)

    with pytest.raises(ValueError, match="no tasks"):
        await planner.plan("some goal")


def test_tester_detect_runner_python(monkeypatch):
    # Change to repo root where pyproject.toml exists
    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(repo_root)

    runner = TesterAgent._detect_runner()
    assert runner == ["pytest", "-q"]