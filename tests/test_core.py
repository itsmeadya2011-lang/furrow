import pytest
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider is not None
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_workspace_validation(tmp_path):
    s = Settings(workspace=tmp_path)
    assert s.workspace == tmp_path


def test_settings_provider_enum():
    from furrow.config import Provider

    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_orchestrator_initialization():
    orch = Orchestrator(goal="test goal")
    assert orch.goal == "test goal"
    assert orch.cycles == 0
    assert orch.plan is None
    assert orch._stop_requested is False


def test_orchestrator_task_results_empty_by_default():
    orch = Orchestrator(goal="test goal")
    assert orch.task_results == {}


def test_orchestrator_request_stop():
    orch = Orchestrator(goal="test goal")
    orch.request_stop()
    assert orch._stop_requested is True


def test_is_done_with_no_plan():
    orch = Orchestrator(goal="test goal")
    assert orch._is_done() is True


def test_is_done_with_all_completed():
    orch = Orchestrator(goal="test goal")
    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ],
        rationale="test",
    )
    assert orch._is_done() is True


def test_is_done_with_failed_task():
    orch = Orchestrator(goal="test goal")
    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ],
        rationale="test",
    )
    assert orch._is_done() is False


def test_is_done_with_pending_tasks():
    orch = Orchestrator(goal="test goal")
    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="pending"),
        ],
        rationale="test",
    )
    assert orch._is_done() is False


def test_task_results_persisted():
    orch = Orchestrator(goal="test goal")
    orch._task_results["1"] = "result1"
    orch._task_results["2"] = "result2"
    assert orch.task_results == {"1": "result1", "2": "result2"}
