import json

import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done_no_tasks():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = []
    assert orch._is_done() is False


def test_orchestrator_is_done_completed():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_orchestrator_is_done_partial():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


def test_orchestrator_is_done_failed():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [
        TaskModel(id="1", description="a", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_get_tasks():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [TaskModel(id="1", description="a")]
    assert orch._get_tasks() == orch.tasks


def test_orchestrator_state_save_and_load(tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    orch.goal = "add feature x"
    orch.cycles = 3
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed", result="done"),
    ]
    orch._state_file = tmp_path / ".furrow_state.json"
    orch._save_state()

    assert orch._state_file.exists()
    raw = json.loads(orch._state_file.read_text())
    assert raw["goal"] == "add feature x"
    assert raw["cycles"] == 3
    assert len(raw["tasks"]) == 1
    assert raw["tasks"][0]["status"] == "completed"

    loaded = Orchestrator.__new__(Orchestrator)
    loaded.goal = "default"
    loaded.cycles = 0
    loaded.tasks = []
    loaded._state_file = tmp_path / ".furrow_state.json"
    loaded._load_state()

    assert loaded.goal == "add feature x"
    assert loaded.cycles == 3
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].id == "1"
    assert loaded.tasks[0].status == "completed"


def test_llm_retryable_connection_error():
    from furrow.llm import _is_retryable

    assert _is_retryable(ConnectionError("boom")) is True


def test_llm_retryable_other_error():
    from furrow.llm import _is_retryable

    assert _is_retryable(ValueError("nope")) is False


def test_llm_retryable_rate_limit_name():
    from furrow.llm import _is_retryable

    class FakeRateLimit(Exception):
        pass

    FakeRateLimit.__name__ = "RateLimitError"
    assert _is_retryable(FakeRateLimit()) is True
