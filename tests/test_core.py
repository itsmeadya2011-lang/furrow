import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_init():
    from furrow.core.orchestrator import Orchestrator
    orch = Orchestrator(goal="test")
    assert orch.goal == "test"
    assert orch.cycles == 0
    assert orch.plan is None


def test_is_done_no_plan():
    from furrow.core.orchestrator import Orchestrator
    orch = Orchestrator(goal="test")
    assert orch._is_done() is False


def test_is_done_with_completed_tasks():
    from furrow.core.orchestrator import Orchestrator
    from furrow.config import Plan, TaskModel
    orch = Orchestrator(goal="test")
    orch.plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")
    orch.plan.tasks[0].status = "completed"
    assert orch._is_done() is True


def test_is_done_with_failed_tasks():
    from furrow.core.orchestrator import Orchestrator
    from furrow.config import Plan, TaskModel
    orch = Orchestrator(goal="test")
    orch.plan = Plan(tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")], rationale="r")
    orch.plan.tasks[0].status = "completed"
    orch.plan.tasks[1].status = "failed"
    assert orch._is_done() is False


def test_max_cycles_setting():
    from furrow.config import Settings
    s = Settings(max_cycles=2)
    assert s.max_cycles == 2


def test_max_parallel_tasks_setting():
    from furrow.config import Settings
    s = Settings(max_parallel_tasks=3)
    assert s.max_parallel_tasks == 3


def test_ollama_provider_enum():
    from furrow.config import Provider
    assert Provider.OLLAMA == "ollama"
    assert Provider.OLLAMA in Provider


@pytest.mark.asyncio
async def test_run_tests_no_runner(tmp_path):
    from furrow.agents.tester import TesterAgent
    from unittest.mock import AsyncMock, patch

    mock_client = AsyncMock()
    mock_client.settings.workspace = tmp_path

    agent = TesterAgent(client=mock_client)

    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        result = await agent._run_tests(workspace=tmp_path)
        assert result == "No test runner found."


@pytest.mark.asyncio
async def test_planner_with_workspace(tmp_path):
    from furrow.agents.planner import PlannerAgent
    from unittest.mock import AsyncMock, Mock

    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "ok"}')
    mock_client.list_files = Mock(return_value=["main.py", "utils.py"])

    agent = PlannerAgent(client=mock_client)
    plan = await agent.plan("test goal", workspace=tmp_path)

    mock_client.list_files.assert_called_once_with(tmp_path)
    assert plan.rationale == "ok"


@pytest.mark.asyncio
async def test_llm_ollama_routing():
    from furrow.llm import LLMClient
    from furrow.config import Provider, Settings
    from unittest.mock import AsyncMock, patch

    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)

    with patch.object(client, "_complete_ollama", new_callable=AsyncMock, return_value="ollama result") as mock_ollama:
        result = await client.complete("test prompt", model="llama3")
        mock_ollama.assert_called_once_with("test prompt", "", "llama3")
        assert result == "ollama result"
