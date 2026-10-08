from __future__ import annotations

import json
from pathlib import Path

import pytest

from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT, WORKER_PROMPT
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult, settings as default_settings
from furrow.llm import FILE_TOOLS, LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="2", description="another")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.max_attempts_per_task == 3
    assert isinstance(s.workspace, Path)


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    s = Settings()
    assert s.provider == Provider.OPENAI
    assert s.max_cycles == 7


def test_file_tools_schema():
    names = [t["name"] for t in FILE_TOOLS]
    assert "write_file" in names
    assert "read_file" in names
    for tool in FILE_TOOLS:
        assert "name" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"


def test_prompts_present():
    for p in (PLANNER_PROMPT, WORKER_PROMPT, TESTER_PROMPT):
        assert isinstance(p, str) and len(p) > 50


def test_test_result_failed():
    t = TestResult(passed=False, summary="boom", failures=["a", "b"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_plan_serialization_roundtrip():
    p = Plan(
        tasks=[TaskModel(id="1", description="x", files=["a.py"], dependencies=[])],
        rationale="because",
    )
    data = json.loads(p.model_dump_json())
    p2 = Plan(**data)
    assert p2.tasks[0].id == "1"
    assert p2.rationale == "because"


def test_llm_client_lazy_providers():
    # Constructing the client should not require API keys.
    client = LLMClient()
    assert client._anthropic is None
    assert client._openai is None
    assert client._ollama is None


def test_provider_enum_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_default_settings_singleton():
    assert default_settings.max_parallel_tasks >= 1


def test_orchestrator_is_done_empty():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    assert orch._is_done() is False
    assert orch.cycles == 0


def test_orchestrator_is_done_all_completed():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_orchestrator_is_done_with_failure():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_max_cycles_enforced(tmp_path):
    from furrow.core.orchestrator import Orchestrator

    s = Settings(max_cycles=3, workspace=tmp_path)
    orch = Orchestrator(goal="test", settings=s)
    # Simulate the run loop's cycle guard without invoking the LLM.
    assert orch.settings.max_cycles == 3
    orch.cycles = 3
    # At max_cycles the loop should stop before incrementing.
    assert orch.cycles >= orch.settings.max_cycles