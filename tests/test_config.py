import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import (
    OrchestratorState,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
)


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test task")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_requires_tasks():
    with pytest.raises(ValidationError):
        Plan(rationale="ok")


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"
    assert s.log_format == "json"


def test_orchestrator_state_save_load():
    state = OrchestratorState(goal="test goal", cycles=3, completed=False)
    state.cycle_states = []

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.json"
        state.save(path)

        loaded = OrchestratorState.load(path)
        assert loaded.goal == "test goal"
        assert loaded.cycles == 3
        assert loaded.completed == False


def test_orchestrator_state_load_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nonexistent.json"
        state = OrchestratorState.load(path)
        assert state.goal == ""
        assert state.cycles == 0


def test_orchestrator_state_json_serialization():
    state = OrchestratorState(goal="g", cycles=1, completed=True)
    data = json.loads(state.model_dump_json())
    assert data["goal"] == "g"
    assert data["cycles"] == 1
    assert data["completed"] is True


def test_settings_env_prefix():
    s = Settings()
    # Just verify it constructs without error
    assert s is not None
