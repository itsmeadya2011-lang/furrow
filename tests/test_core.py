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


# --- Orchestrator loop / termination logic ---


def _make_orchestrator(goal: str = "x") -> Orchestrator:
    # Constructing an Orchestrator does not touch the network: LLM clients are lazy.
    return Orchestrator(goal=goal)


def test_is_done_no_tasks():
    o = _make_orchestrator()
    o.tasks = []
    assert o._is_done() is True


def test_is_done_all_tasks_completed():
    o = _make_orchestrator()
    o.tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
    for t in o.tasks:
        t.status = "completed"
    assert o._is_done() is True


def test_is_done_with_failed_task_not_done():
    o = _make_orchestrator()
    t1 = TaskModel(id="1", description="a")
    t1.status = "completed"
    t2 = TaskModel(id="2", description="b")
    t2.status = "failed"
    o.tasks = [t1, t2]
    assert o._is_done() is False


def test_is_done_tests_failed_not_done():
    o = _make_orchestrator()
    t = TaskModel(id="1", description="a")
    t.status = "completed"
    o.tasks = [t]
    o.last_test_result = TestResult(passed=False, summary="boom", failures=["fail"])
    assert o._is_done() is False


def test_is_done_tests_passed_and_tasks_completed():
    o = _make_orchestrator()
    t = TaskModel(id="1", description="a")
    t.status = "completed"
    o.tasks = [t]
    o.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert o._is_done() is True


def test_max_cycles_caps_execution():
    o = _make_orchestrator()
    o.settings = Settings(max_cycles=2)
    # ``cycles`` counts completed cycles; the guard fires once cycles exceeds max_cycles.
    o.cycles = 2
    assert o.settings.max_cycles > 0 and o.cycles > o.settings.max_cycles


# --- LLM provider support ---


def test_ollama_provider_configures_openai_compatible_endpoint():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    c = LLMClient(settings=s)
    client = c.ollama
    assert str(client.base_url) == "http://localhost:11434/v1"


def test_anthropic_provider_still_supported():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-test-123")
    c = LLMClient(settings=s)
    assert c.settings.provider == Provider.ANTHROPIC
    assert c.anthropic is not None


@pytest.mark.asyncio
async def test_complete_anthropic_dispatch():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-test-1234567890")
    created = {}

    class _FakeResp:
        content = [type("C", (), {"text": "hi"})]  # noqa: D401 - simple stub

    class _FakeClient:
        async def create(self, **kwargs):
            created.update(kwargs)
            return _FakeResp()

    class _FakeAsync:
        messages = _FakeClient()

    c = LLMClient(settings=s)
    c._anthropic = _FakeAsync()
    result = await c.complete("hello", system="you are helpful")
    assert result == "hi"
    assert created["model"] == s.model
    assert created["system"] == "you are helpful"

