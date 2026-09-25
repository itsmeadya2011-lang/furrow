import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator, OrchestratorState
from furrow.llm import LLMClient, MODEL_COSTS


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test")
    assert task.status == "pending"
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_settings_defaults():
    s = Settings()
    assert s.provider == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_settings_provider_models():
    s = Settings(provider="openai")
    assert s.get_model_for_agent("planner") == "claude-3-5-haiku-20241022"
    assert s.get_model_for_agent("worker") == "claude-3-5-sonnet-20241022"
    assert s.get_model_for_agent("tester") == "claude-3-5-sonnet-20241022"
    assert s.get_model_for_agent("unknown") == s.model


def test_orchestrator_state():
    state = OrchestratorState(
        original_goal="test goal",
        current_goal="test goal",
        cycle=1,
    )
    assert state.original_goal == "test goal"
    assert state.cycle == 1
    assert state.all_tasks == []


def test_llm_client_cost_tracking():
    client = LLMClient()
    # Simulate tracking
    client._track_usage("claude-3-5-sonnet-20241022", 1000, 500)
    summary = client.get_cost_summary()
    
    # claude-3-5-sonnet: input=$0.003/1k, output=$0.015/1k
    # 1000 input = $0.003, 500 output = $0.0075
    assert summary["total_input_tokens"] == 1000
    assert summary["total_output_tokens"] == 500
    assert abs(summary["total_cost_usd"] - 0.0105) < 0.0001


def test_model_costs_exist():
    assert "claude-3-5-sonnet-20241022" in MODEL_COSTS
    assert "gpt-4o" in MODEL_COSTS
    assert MODEL_COSTS["claude-3-5-sonnet-20241022"]["input"] > 0


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    orchestrator = Orchestrator(goal="test goal")
    assert orchestrator.state.original_goal == "test goal"
    assert orchestrator.state.current_goal == "test goal"
    assert orchestrator.state.cycle == 0


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    settings = Settings(max_cycles=2)
    orchestrator = Orchestrator(goal="test goal", settings=settings)
    assert orchestrator._max_cycles == 2


def test_task_model_dependencies():
    task = TaskModel(
        id="1",
        description="task with deps",
        dependencies=["2", "3"]
    )
    assert task.dependencies == ["2", "3"]


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
        TaskModel(id="3", description="task 3"),
    ]
    plan = Plan(tasks=tasks, rationale="test plan")
    assert len(plan.tasks) == 3
    assert plan.tasks[1].dependencies == ["1"]


def test_test_result_failures():
    result = TestResult(
        passed=False,
        summary="tests failed",
        failures=["test_foo failed", "test_bar failed"]
    )
    assert result.passed is False
    assert len(result.failures) == 2


@pytest.mark.asyncio
async def test_llm_client_complete_mock():
    with patch("furrow.llm.AsyncAnthropic") as mock_anthropic:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="test response")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client
        
        client = LLMClient(Settings(provider="anthropic", anthropic_api_key="test"))
        result = await client.complete("test prompt")
        
        assert result == "test response"
        assert client.total_input_tokens == 100
        assert client.total_output_tokens == 50


@pytest.mark.asyncio
async def test_llm_client_ollama():
    with patch("furrow.llm.ollama.AsyncClient") as mock_ollama:
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": "ollama response"}})
        mock_ollama.return_value = mock_client
        
        client = LLMClient(Settings(provider="ollama"))
        result = await client.complete("test prompt", model="llama3.1")
        
        assert result == "ollama response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])