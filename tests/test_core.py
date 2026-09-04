import json
from pathlib import Path

import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
        ],
        rationale="ok",
    )
    orchestrator = Orchestrator(goal="test")
    orchestrator.plan = plan
    assert len(orchestrator._get_tasks()) == 2


def test_orchestrator_is_done_no_tasks():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_all_completed():
    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1", status="completed")],
        rationale="ok",
    )
    orchestrator = Orchestrator(goal="test")
    orchestrator.plan = plan
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_has_failed():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ],
        rationale="ok",
    )
    orchestrator = Orchestrator(goal="test")
    orchestrator.plan = plan
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_all_pending():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
        ],
        rationale="ok",
    )
    orchestrator = Orchestrator(goal="test")
    orchestrator.plan = plan
    assert orchestrator._is_done() is False


def test_orchestrator_state_persistence(tmp_path: Path):
    state_file = tmp_path / ".furrow" / "state.json"
    orchestrator = Orchestrator(goal="original goal")
    orchestrator.state_file = state_file
    orchestrator.cycles = 3
    orchestrator.goal = "updated goal"
    orchestrator._save_state()

    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["cycles"] == 3
    assert data["goal"] == "updated goal"


def test_orchestrator_state_load(tmp_path: Path):
    state_file = tmp_path / ".furrow" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(json.dumps({"cycles": 5, "goal": "loaded goal"}))

    orchestrator = Orchestrator(goal="new goal")
    orchestrator.state_file = state_file
    orchestrator._load_state()

    assert orchestrator.cycles == 5
    assert orchestrator.goal == "loaded goal"


def test_orchestrator_state_load_missing_file():
    orchestrator = Orchestrator(goal="goal")
    orchestrator.state_file = Path("/nonexistent/path/state.json")
    orchestrator._load_state()
    assert orchestrator.cycles == 0
    assert orchestrator.goal == "goal"


def test_orchestrator_state_load_invalid_json(tmp_path: Path):
    state_file = tmp_path / ".furrow" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text("not json")

    orchestrator = Orchestrator(goal="goal")
    orchestrator.state_file = state_file
    orchestrator._load_state()
    assert orchestrator.cycles == 0
    assert orchestrator.goal == "goal"
