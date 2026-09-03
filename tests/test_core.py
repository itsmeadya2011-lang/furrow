import json
from pathlib import Path

from furrow.config import (
    Plan,
    Settings,
    State,
    TaskModel,
    TestResult,
)
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_state_round_trip():
    state = State(
        goal="build a thing",
        cycles=3,
        tasks=[
            TaskModel(id="1", description="first", status="completed", result="done"),
            TaskModel(id="2", description="second", status="pending"),
        ],
        last_test_passed=True,
        last_failures=[],
        updated_at="2026-01-01T00:00:00+00:00",
    )

    raw = json.loads(json.dumps(state.model_dump()))
    restored = State.model_validate(raw)

    assert restored.goal == state.goal
    assert restored.cycles == state.cycles
    assert len(restored.tasks) == 2
    assert restored.tasks[0].id == "1"
    assert restored.tasks[0].status == "completed"
    assert restored.tasks[1].status == "pending"
    assert restored.last_test_passed is True
    assert restored.last_failures == []
    assert restored.updated_at == state.updated_at


def test_state_defaults():
    state = State(
        goal="g",
        cycles=0,
        tasks=[],
        updated_at="2026-01-01T00:00:00+00:00",
    )
    assert state.last_test_passed is None
    assert state.last_failures == []


def test_settings_new_field_defaults():
    s = Settings(_env_file=None)
    assert s.state_file == Path.cwd() / ".furrow_state.json"
    assert s.max_tokens == 4096
    assert s.max_consecutive_failures == 3


def _make_orchestrator(tmp_path: Path) -> Orchestrator:
    settings = Settings(_env_file=None, state_file=tmp_path / "state.json")
    return Orchestrator(goal="test goal", client=type("C", (), {"settings": settings})(), settings=settings)


def test_is_done_all_completed_and_tests_passed(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orch.last_test_passed = True
    assert orch._is_done() is True


def test_is_done_pending_task(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    orch.last_test_passed = True
    assert orch._is_done() is False


def test_is_done_tests_failed(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
    ]
    orch.last_test_passed = False
    assert orch._is_done() is False


def test_is_done_no_tasks_and_tests_passed(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.tasks = []
    orch.last_test_passed = True
    assert orch._is_done() is True


def test_is_done_no_tasks_and_tests_failed(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.tasks = []
    orch.last_test_passed = False
    assert orch._is_done() is False


def test_is_done_task_failed(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    orch.last_test_passed = True
    assert orch._is_done() is False


def test_save_and_load_state(tmp_path):
    settings = Settings(_env_file=None, state_file=tmp_path / "state.json")
    orch = Orchestrator(
        goal="orig goal",
        client=type("C", (), {"settings": settings})(),
        settings=settings,
    )
    orch.cycles = 2
    orch.tasks = [TaskModel(id="1", description="a", status="completed", result="r")]
    orch.last_test_passed = True
    orch.last_failures = []
    orch.save_state()

    assert settings.state_file.exists()

    settings2 = Settings(_env_file=None, state_file=tmp_path / "state.json")
    orch2 = Orchestrator(
        goal="orig goal",
        client=type("C", (), {"settings": settings2})(),
        settings=settings2,
    )
    assert orch2.cycles == 2
    assert len(orch2.tasks) == 1
    assert orch2.tasks[0].id == "1"
    assert orch2.tasks[0].status == "completed"
    assert orch2.last_test_passed is True


def test_load_state_missing_file(tmp_path):
    settings = Settings(_env_file=None, state_file=tmp_path / "missing.json")
    orch = Orchestrator(
        goal="g",
        client=type("C", (), {"settings": settings})(),
        settings=settings,
    )
    assert orch.cycles == 0
    assert orch.tasks == []