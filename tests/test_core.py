import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_requires_tasks_and_rationale():
    with pytest.raises(Exception):
        Plan(tasks=[], rationale="ok")
    with pytest.raises(Exception):
        Plan(tasks=[TaskModel(id="1", description="x")], rationale="")


def test_test_result_defaults():
    t = TestResult(passed=False, summary="fail")
    assert t.failures == []


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.planner_model == "claude-3-5-haiku-20241022"
    assert s.worker_model == "claude-3-5-sonnet-20241022"
    assert s.tester_model == "claude-3-5-sonnet-20241022"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_orchestrator_is_done_no_tasks():
    from furrow.core.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o._current_plan = None
    assert o._is_done() is False


def test_orchestrator_is_done_with_completed_tasks():
    from furrow.core.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    assert o._is_done() is True


def test_orchestrator_is_done_with_failed_tasks():
    from furrow.core.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="ok",
    )
    assert o._is_done() is False


def test_orchestrator_is_done_with_pending_tasks():
    from furrow.core.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ],
        rationale="ok",
    )
    assert o._is_done() is False


def test_orchestrator_max_cycles_enforcement():
    from furrow.core.orchestrator import Orchestrator
    from furrow.config import Settings

    settings = Settings(max_cycles=2)
    o = Orchestrator(goal="test", settings=settings)
    assert o.settings.max_cycles == 2
    assert o.cycles == 0
    # We don't run the full loop here, just verify the setting is wired


def test_llm_client_providers():
    from furrow.llm import LLMClient
    from furrow.config import Settings, Provider

    s = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=s)
    assert client.settings.provider == Provider.ANTHROPIC

    s2 = Settings(provider=Provider.OPENAI)
    client2 = LLMClient(settings=s2)
    assert client2.settings.provider == Provider.OPENAI


def test_llm_client_ollama():
    from furrow.llm import LLMClient
    from furrow.config import Settings, Provider

    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)
    assert client.settings.provider == Provider.OLLAMA
