import pytest
from furrow.config import Plan, TaskModel, TaskStatus, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_status_enum():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"
    assert TaskStatus.IN_PROGRESS == "in_progress"


def test_task_status_is_string_enum():
    assert isinstance(TaskStatus.PENDING, str)
    assert isinstance(TaskStatus.PENDING, TaskStatus)


def test_task_model_default_status():
    task = TaskModel(id="1", description="test")
    assert task.status == TaskStatus.PENDING


def test_task_model_with_status():
    task = TaskModel(id="1", description="test", status=TaskStatus.COMPLETED)
    assert task.status == TaskStatus.COMPLETED


def test_orchestrator_initial_state():
    from furrow.core.orchestrator import Orchestrator
    from furrow.llm import LLMClient

    client = LLMClient.__new__(LLMClient)
    orchestrator = Orchestrator(goal="test goal", client=client)
    assert orchestrator.goal == "test goal"
    assert orchestrator.cycles == 0
    assert orchestrator.plan is None
    assert orchestrator.tasks == []
    # Fresh orchestrator with no tasks is not done (hasn't planned yet)
    assert not orchestrator._is_done()
    assert not orchestrator._exceeded_max_cycles()


def test_orchestrator_done_after_no_tasks():
    from furrow.core.orchestrator import Orchestrator
    from furrow.llm import LLMClient
    from furrow.config import TaskStatus

    client = LLMClient.__new__(LLMClient)
    orchestrator = Orchestrator(goal="test goal", client=client)
    # Simulate a cycle that planned but found no tasks
    orchestrator._all_tasks_completed = True
    assert orchestrator._is_done() is True


def test_orchestrator_max_cycles_check():
    from furrow.core.orchestrator import Orchestrator
    from furrow.llm import LLMClient
    from furrow.config import TaskStatus

    client = LLMClient.__new__(LLMClient)
    orchestrator = Orchestrator(goal="test goal", client=client)
    orchestrator.cycles = 4
    from furrow.config import settings
    original_max = settings.max_cycles
    settings.max_cycles = 5
    assert orchestrator._exceeded_max_cycles() is False
    orchestrator.cycles = 5
    assert orchestrator._exceeded_max_cycles() is True
    settings.max_cycles = original_max


def test_orchestrator_done_with_completed_tasks():
    from furrow.core.orchestrator import Orchestrator
    from furrow.llm import LLMClient
    from furrow.config import TaskModel, TaskStatus

    client = LLMClient.__new__(LLMClient)
    orchestrator = Orchestrator(goal="test goal", client=client)
    orchestrator._tasks = [TaskModel(id="1", description="test", status=TaskStatus.COMPLETED)]
    assert orchestrator._is_done() is True


def test_orchestrator_not_done_with_pending_tasks():
    from furrow.core.orchestrator import Orchestrator
    from furrow.llm import LLMClient
    from furrow.config import TaskModel, TaskStatus

    client = LLMClient.__new__(LLMClient)
    orchestrator = Orchestrator(goal="test goal", client=client)
    orchestrator._tasks = [TaskModel(id="1", description="test", status=TaskStatus.PENDING)]
    assert orchestrator._is_done() is False


def test_orchestrator_not_done_with_failed_tasks():
    from furrow.core.orchestrator import Orchestrator
    from furrow.llm import LLMClient
    from furrow.config import TaskModel, TaskStatus

    client = LLMClient.__new__(LLMClient)
    orchestrator = Orchestrator(goal="test goal", client=client)
    orchestrator._tasks = [TaskModel(id="1", description="test", status=TaskStatus.FAILED)]
    assert orchestrator._is_done() is False


def test_settings_log_level_validation():
    from furrow.config import Settings
    s = Settings(log_level="debug")
    assert s.log_level == "DEBUG"
    s2 = Settings(log_level="ERROR")
    assert s2.log_level == "ERROR"


def test_settings_max_parallel_tasks_validation():
    from furrow.config import Settings
    s = Settings(max_parallel_tasks=3)
    assert s.max_parallel_tasks == 3
    with pytest.raises(Exception):
        Settings(max_parallel_tasks=0)


def test_setup_logging():
    from furrow.config import setup_logging
    setup_logging("DEBUG")
    import logging
    root = logging.getLogger()
    assert root.level == logging.DEBUG


def test_settings_validation_with_structlog():
    # Verify structlog import works in config
    from furrow.config import setup_logging, settings
    setup_logging()
    assert settings is not None
