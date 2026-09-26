import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_defaults():
    t = TaskModel(id="x", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_with_dependencies():
    t1 = TaskModel(id="1", description="first")
    t2 = TaskModel(id="2", description="second", dependencies=["1"])
    p = Plan(tasks=[t1, t2], rationale="build order")
    assert p.tasks[1].dependencies == ["1"]


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.ollama_base_url == "http://localhost:11434"


def test_settings_provider_override():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:8080")
    assert s.provider == Provider.OLLAMA
    assert s.ollama_base_url == "http://localhost:8080"


def test_settings_max_parallel_override():
    s = Settings(max_parallel_tasks=10, max_cycles=3)
    assert s.max_parallel_tasks == 10
    assert s.max_cycles == 3


def test_settings_model_fields():
    s = Settings(
        provider=Provider.OPENAI,
        model="gpt-4",
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test",
    )
    assert s.provider == Provider.OPENAI
    assert s.model == "gpt-4"
    assert s.openai_api_key == "sk-test"
    assert s.anthropic_api_key == "sk-ant-test"


@pytest.mark.asyncio
async def test_llm_client_unsupported_provider():
    """LLMClient should raise ValueError for unsupported provider."""
    from furrow.llm import LLMClient

    # Set provider to a non-standard value (bypasses validation since
    # validate_assignment defaults to False in pydantic v2)
    s = Settings()
    s.provider = "custom"
    client = LLMClient(settings=s)

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test prompt")
