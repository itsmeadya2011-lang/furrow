import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, Settings, TaskModel, TestResult, Provider
from furrow.llm import LLMClient
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test task")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0


@pytest.mark.asyncio
async def test_llm_client_complete_anthropic():
    with patch("furrow.llm.AsyncAnthropic") as mock_anthropic:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client

        client = LLMClient(Settings(anthropic_api_key="test-key"))
        result = await client.complete("Test prompt", system="Test system")

        assert result == "Test response"
        mock_client.messages.create.assert_called_once()


@pytest.mark.asyncio
async def test_llm_client_complete_openai():
    with patch("furrow.llm.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        client = LLMClient(Settings(provider=Provider.OPENAI, openai_api_key="test-key"))
        result = await client.complete("Test prompt", system="Test system")

        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_planner_agent():
    with patch("furrow.llm.LLMClient.complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = '{"tasks": [{"id": "1", "description": "Test task", "files": [], "dependencies": []}], "rationale": "Test rationale"}'

        client = LLMClient(Settings(anthropic_api_key="test-key"))
        planner = PlannerAgent(client=client)
        plan = await planner.plan("Test goal")

        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "Test task"
        assert plan.rationale == "Test rationale"


@pytest.mark.asyncio
async def test_worker_agent():
    with patch("furrow.llm.LLMClient.complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = "Task completed successfully"

        client = LLMClient(Settings(anthropic_api_key="test-key"))
        task = TaskModel(id="1", description="Test task")
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()

        assert result == "Task completed successfully"


@pytest.mark.asyncio
async def test_tester_agent():
    with patch("furrow.llm.LLMClient.complete", new_callable=AsyncMock) as mock_complete:
        with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "test output"
            mock_complete.return_value = '{"passed": true, "summary": "All tests passed", "failures": []}'

            client = LLMClient(Settings(anthropic_api_key="test-key"))
            tester = TesterAgent(client=client)
            tasks = [TaskModel(id="1", description="Test task", status="completed")]
            result = await tester.run("Test goal", tasks)

            assert result.passed is True
            assert result.summary == "All tests passed"


@pytest.mark.asyncio
async def test_orchestrator_cycle():
    with patch("furrow.llm.LLMClient.complete", new_callable=AsyncMock) as mock_complete:
        with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "test output"
            # First call: planner response
            # Second call: worker response
            # Third call: tester response
            mock_complete.side_effect = [
                '{"tasks": [{"id": "1", "description": "Test task", "files": [], "dependencies": []}], "rationale": "Test rationale"}',
                "Task completed successfully",
                '{"passed": true, "summary": "All tests passed", "failures": []}',
            ]

            client = LLMClient(Settings(anthropic_api_key="test-key", max_cycles=1))
            orchestrator = Orchestrator(goal="Test goal", client=client)
            await orchestrator.run()

            assert orchestrator.cycles == 1
            assert len(orchestrator.all_tasks) == 1
            assert orchestrator.all_tasks[0].status == "completed"


def test_orchestrator_state_persistence(tmp_path):
    state_file = tmp_path / ".furrow_state.json"
    state_file.write_text('{"cycles": 2, "original_goal": "Original goal", "goal": "Current goal", "all_tasks": [{"id": "1", "description": "Task 1", "files": [], "dependencies": [], "status": "completed", "result": "Done"}]}')

    with patch("furrow.llm.LLMClient.complete", new_callable=AsyncMock) as mock_complete:
        with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "test output"
            mock_complete.return_value = '{"passed": true, "summary": "All tests passed", "failures": []}'

            client = LLMClient(Settings(anthropic_api_key="test-key", workspace=tmp_path, max_cycles=1))
            orchestrator = Orchestrator(goal="Test goal", client=client)

            assert orchestrator.cycles == 2
            assert orchestrator.original_goal == "Original goal"
            assert orchestrator.goal == "Current goal"
            assert len(orchestrator.all_tasks) == 1
            assert orchestrator.all_tasks[0].status == "completed"
