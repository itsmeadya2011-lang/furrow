import json
from pathlib import Path

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT, WORKER_PROMPT
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="x")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_env_prefix(monkeypatch):
    monkeypatch.setenv("FURROW_MODEL", "test-model")
    s = Settings()
    assert s.model == "test-model"


def test_prompts_are_non_empty():
    assert len(PLANNER_PROMPT) > 100
    assert len(WORKER_PROMPT) > 100
    assert len(TESTER_PROMPT) > 100


def test_prompts_contain_json_instruction():
    assert "JSON" in PLANNER_PROMPT
    assert "JSON" in TESTER_PROMPT


@pytest.mark.asyncio
async def test_planner_agent_instantiation():
    agent = PlannerAgent()
    assert agent.client is not None


@pytest.mark.asyncio
async def test_worker_agent_instantiation():
    task = TaskModel(id="1", description="do thing")
    agent = WorkerAgent(task=task)
    assert agent.task.id == "1"


@pytest.mark.asyncio
async def test_tester_agent_instantiation():
    agent = TesterAgent()
    assert agent.client is not None


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    orch = Orchestrator(goal="test")
    assert orch.goal == "test"
    assert orch.cycles == 0
    assert orch._last_tasks == []


@pytest.mark.asyncio
async def test_orchestrator_is_done_empty_tasks():
    orch = Orchestrator(goal="test")
    assert orch._is_done() is True


def test_llm_client_list_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")

    client = LLMClient()
    files = client.list_files(str(tmp_path))
    assert "a.txt" in files
    assert "b.txt" in files
    assert "sub/c.txt" in files


def test_llm_client_list_files_missing():
    client = LLMClient()
    assert client.list_files("/nonexistent/path/12345") == []


def test_test_result_json_roundtrip():
    data = {
        "passed": False,
        "summary": "2 failed",
        "failures": ["assert x == y", "assert a == b"],
    }
    t = TestResult(**data)
    assert t.passed is False
    assert len(t.failures) == 2
    assert t.model_dump() == data


def test_plan_json_roundtrip():
    data = {
        "tasks": [
            {
                "id": "1",
                "description": "do thing",
                "files": ["a.py"],
                "dependencies": [],
            }
        ],
        "rationale": "ok",
    }
    p = Plan(**data)
    assert p.tasks[0].id == "1"
    assert p.tasks[0].files == ["a.py"]
    assert p.model_dump() == data
