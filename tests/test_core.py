import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_task_model_with_all_fields():
    task = TaskModel(
        id="1",
        description="do thing",
        files=["a.py", "b.py"],
        dependencies=["0"],
        status="pending",
        result=None,
    )
    assert task.id == "1"
    assert task.description == "do thing"
    assert task.files == ["a.py", "b.py"]
    assert task.dependencies == ["0"]
    assert task.status == "pending"
    assert task.result is None


def test_task_model_defaults():
    task = TaskModel(id="1", description="do thing")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="first"),
        TaskModel(id="2", description="second"),
        TaskModel(id="3", description="third"),
    ]
    plan = Plan(tasks=tasks, rationale="build feature")
    assert len(plan.tasks) == 3
    assert plan.tasks[0].description == "first"
    assert plan.tasks[1].description == "second"
    assert plan.tasks[2].description == "third"
    assert plan.rationale == "build feature"


def test_test_result_passed():
    result = TestResult(passed=True, summary="all good")
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


def test_test_result_failed():
    result = TestResult(passed=False, summary="something broke", failures=["missing import", "type error"])
    assert result.passed is False
    assert result.summary == "something broke"
    assert result.failures == ["missing import", "type error"]


def test_plan_validates_task_structure():
    with pytest.raises(ValidationError):
        Plan(tasks="not a list", rationale="bad")  # type: ignore[arg-type]


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == "anthropic"
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.planner_model == "claude-3-5-haiku-20241022"
    assert settings.worker_model == "claude-3-5-sonnet-20241022"
    assert settings.tester_model == "claude-3-5-sonnet-20241022"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.log_level == "INFO"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MODEL", "gpt-4")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "10")
    settings = Settings()
    assert settings.provider == "openai"
    assert settings.model == "gpt-4"
    assert settings.max_parallel_tasks == 10


def test_orchestrator_initialization():
    mock_client = MagicMock(spec=LLMClient)
    orchestrator = Orchestrator(goal="build a thing", client=mock_client)
    assert orchestrator.goal == "build a thing"
    assert orchestrator.client is mock_client
    assert orchestrator.cycles == 0
    assert orchestrator.task_results == {}


def test_orchestrator_state_save_load(tmp_path):
    state_file = tmp_path / "state.json"
    mock_client = MagicMock(spec=LLMClient)
    orchestrator = Orchestrator(
        goal="build a thing",
        client=mock_client,
        state_file=str(state_file),
    )
    orchestrator.cycles = 2
    orchestrator.task_results = {"task-1": "done"}
    orchestrator._save_state()

    assert state_file.exists()
    with state_file.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["goal"] == "build a thing"
    assert raw["cycles"] == 2
    assert raw["task_results"] == {"task-1": "done"}
    assert "last_updated" in raw

    loaded = orchestrator._load_state()
    assert loaded is not None
    assert loaded["goal"] == "build a thing"
    assert loaded["cycles"] == 2
    assert loaded["task_results"] == {"task-1": "done"}


def test_empty_task_list():
    plan = Plan(tasks=[], rationale="nothing to do")
    assert plan.tasks == []


def test_task_with_dependencies():
    task = TaskModel(id="2", description="depends on 1", dependencies=["1"])
    assert task.dependencies == ["1"]
    plan = Plan(tasks=[task], rationale="chain")
    assert plan.tasks[0].dependencies == ["1"]
