import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# Task / plan model
# ---------------------------------------------------------------------------


def test_task_model_defaults():
    t = TaskModel(id="1", description="x")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_plan_with_multiple_tasks():
    tasks = [TaskModel(id=str(i), description=f"t{i}") for i in range(3)]
    p = Plan(tasks=tasks, rationale="r")
    assert len(p.tasks) == 3


# ---------------------------------------------------------------------------
# Settings / provider
# ---------------------------------------------------------------------------


def test_provider_enum_values():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5


def test_settings_max_cycles_override():
    s = Settings(max_cycles=2)
    assert s.max_cycles == 2


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def test_llm_tool_schemas():
    client = LLMClient(settings=Settings())
    names = {t["name"] for t in client._tool_schemas()}
    assert names == {"read_file", "write_file", "list_files"}


def test_ollama_client_base_url_normalized():
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)
    # The OpenAI client normalizes the URL with a trailing slash; compare
    # path-stripped to stay robust across httpx versions.
    assert str(client.ollama.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_safe_path_workspace_confinement(tmp_path):
    s = Settings(workspace=tmp_path)
    client = LLMClient(settings=s)
    within = client._safe_path("foo.txt")
    assert within == (tmp_path / "foo.txt").resolve()

    absolute_inside = client._safe_path(str(tmp_path / "bar.txt"))
    assert absolute_inside == (tmp_path / "bar.txt").resolve()

    with pytest.raises(PermissionError):
        client._safe_path("/etc/passwd")


# ---------------------------------------------------------------------------
# Orchestrator logic (no LLM calls)
# ---------------------------------------------------------------------------


def _orchestrator():
    return Orchestrator(goal="g")


def test_orchestrator_is_done_when_all_completed_and_tests_pass():
    o = _orchestrator()
    o.tasks = [TaskModel(id="1", description="d", status="completed")]
    o.last_test_result = TestResult(passed=True, summary="ok")
    assert o._is_done() is True


def test_orchestrator_not_done_when_a_task_failed():
    o = _orchestrator()
    o.tasks = [TaskModel(id="1", description="d", status="failed")]
    o.last_test_result = TestResult(passed=True, summary="ok")
    assert o._is_done() is False


def test_orchestrator_not_done_when_tests_fail():
    o = _orchestrator()
    o.tasks = [TaskModel(id="1", description="d", status="completed")]
    o.last_test_result = TestResult(passed=False, summary="bad", failures=["x"])
    assert o._is_done() is False


def test_orchestrator_not_done_when_no_tasks_yet():
    o = _orchestrator()
    assert o._is_done() is False


def test_orchestrator_respects_max_cycles():
    o = _orchestrator()
    o.client.settings.max_cycles = 1
    o.cycles = 1
    # Even with nothing done, hitting max_cycles stops the loop.
    assert o._is_done() is True


def test_orchestrator_no_tasks_cycle_is_done():
    # After a cycle where the planner returned no tasks, the goal is complete.
    o = _orchestrator()
    o.tasks = []
    o.last_test_result = TestResult(passed=True, summary="No tasks remaining")
    assert o._is_done() is True


def test_orchestrator_get_tasks_returns_tracked():
    o = _orchestrator()
    o.tasks = [TaskModel(id="1", description="d", status="completed")]
    assert o._get_tasks() is o.tasks
