import pytest
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5


def test_ollama_provider_enum():
    assert Provider.OLLAMA == "ollama"


def test_llm_client_ollama_model_selection():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)
    assert client.settings.provider == Provider.OLLAMA


def test_llm_client_unsupported_provider():
    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.provider = "bad_provider"
    mock_settings.model = "test-model"
    mock_settings.anthropic_api_key = None
    mock_settings.openai_api_key = None

    client = LLMClient(settings=mock_settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio_run(client.complete("test"))


def asyncio_run(coro):
    import asyncio

    try:
        return asyncio.get_event_loop().run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def test_orchestrator_is_done_empty():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    assert orch._is_done() is False


def test_orchestrator_is_done_completed():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    orch.all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_orchestrator_is_done_with_failure():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    orch.all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_max_cycles():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    orch.cycles = 10
    assert orch._is_done() is False
    # max_cycles check is in run(), not _is_done(), but we verify the field is accessible
    assert orch.client.settings.max_cycles == 0
