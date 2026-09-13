import pytest
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_default_settings():
    orb = Orchestrator(goal="test goal")
    assert orb.goal == "test goal"
    assert orb.settings is not None
    assert orb.cycles == 0
    assert orb.tasks == []
    assert orb.output_callback is None
    assert orb.logger is not None


def test_orchestrator_with_settings():
    settings = Settings(max_cycles=5, max_parallel_tasks=3, log_level="DEBUG")
    orb = Orchestrator(goal="test", settings=settings)
    assert orb.settings.max_cycles == 5
    assert orb.settings.max_parallel_tasks == 3


def test_orchestrator_with_output_callback():
    messages: list[str] = []

    def callback(msg: str) -> None:
        messages.append(msg)

    orb = Orchestrator(goal="test", output_callback=callback)
    assert orb.output_callback is callback


def test_is_done_with_no_tasks():
    orb = Orchestrator(goal="test")
    assert orb._is_done() is True


def test_is_done_with_all_completed():
    orb = Orchestrator(goal="test")
    orb.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orb._is_done() is True


def test_is_done_with_failed_task():
    orb = Orchestrator(goal="test")
    orb.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orb._is_done() is False


def test_is_done_with_pending_tasks():
    orb = Orchestrator(goal="test")
    orb.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert orb._is_done() is False


def test_settings_max_cycles_zero_unlimited():
    settings = Settings(max_cycles=0)
    assert settings.max_cycles == 0


def test_settings_max_parallel_tasks_default():
    settings = Settings()
    assert settings.max_parallel_tasks == 5
    assert settings.log_level == "INFO"
