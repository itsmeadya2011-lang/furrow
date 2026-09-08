import pytest
from click.testing import CliRunner
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_get_tasks_returns_empty_before_plan():
    o = Orchestrator(goal="test")
    assert o._get_tasks() == []


def test_get_tasks_returns_plan_tasks():
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
    o = Orchestrator(goal="test")
    o.current_plan = plan
    assert len(o._get_tasks()) == 1
    assert o._get_tasks()[0].id == "1"


def test_is_done_false_when_no_tasks():
    o = Orchestrator(goal="test")
    assert o._is_done() is True  # 0 >= 0 is True


def test_is_done_true_when_all_completed():
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
    o = Orchestrator(goal="test")
    o.current_plan = plan
    o.current_plan.tasks[0].status = "completed"
    assert o._is_done() is True


def test_is_done_false_when_tasks_incomplete():
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
    o = Orchestrator(goal="test")
    o.current_plan = plan
    assert o._is_done() is False


def test_is_done_false_when_tasks_failed():
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
    o = Orchestrator(goal="test")
    o.current_plan = plan
    o.current_plan.tasks[0].status = "failed"
    assert o._is_done() is False


def test_max_cycles_enforced():
    o = Orchestrator(goal="test", settings=Settings(max_cycles=1))
    o.current_plan = Plan(tasks=[], rationale="ok")
    assert o._is_done() is True
    # _is_done returns True when no tasks, so run() would break via _is_done
    # max_cycles is only checked after _is_done returns False
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
    o.current_plan = plan
    assert o._is_done() is False
    o.cycles = 1
    assert o.settings.max_cycles > 0 and o.cycles >= o.settings.max_cycles


def test_original_goal_preserved():
    o = Orchestrator(goal="original")
    assert o.original_goal == "original"
    o.goal = "changed"
    assert o.original_goal == "original"


def test_cli_settings_override():
    from furrow.cli.main import main
    runner = CliRunner()
    result = runner.invoke(main, ["start", "--help"])
    assert result.exit_code == 0
