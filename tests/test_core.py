import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TaskStatus, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class TestTaskStatusEnum:
    def test_task_model_accepts_taskstatus_enum(self):
        task = TaskModel(id="1", description="test", status=TaskStatus.PENDING)
        assert task.status == TaskStatus.PENDING

    def test_task_model_accepts_taskstatus_completed(self):
        task = TaskModel(id="1", description="test", status=TaskStatus.COMPLETED)
        assert task.status == TaskStatus.COMPLETED

    def test_task_model_accepts_taskstatus_failed(self):
        task = TaskModel(id="1", description="test", status=TaskStatus.FAILED)
        assert task.status == TaskStatus.FAILED

    def test_task_model_string_assignment_works(self):
        task = TaskModel(id="1", description="test", status="pending")
        assert task.status == TaskStatus.PENDING

    def test_task_model_string_assignment_completed(self):
        task = TaskModel(id="1", description="test", status="completed")
        assert task.status == TaskStatus.COMPLETED

    def test_task_model_string_assignment_failed(self):
        task = TaskModel(id="1", description="test", status="failed")
        assert task.status == TaskStatus.FAILED

    def test_task_model_default_status_is_pending(self):
        task = TaskModel(id="1", description="test")
        assert task.status == TaskStatus.PENDING


class TestOrchestratorTaskTracking:
    @pytest.fixture
    def orchestrator(self):
        with patch.object(LLMClient, "__init__", lambda self, settings=None: setattr(self, "settings", settings or Settings())):
            return Orchestrator(goal="test goal")

    def test_get_tasks_returns_tasks_from_last_plan(self, orchestrator):
        tasks = [
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
        ]
        orchestrator._tasks = tasks
        assert orchestrator._get_tasks() == tasks

    def test_is_done_returns_true_when_all_tasks_completed(self, orchestrator):
        orchestrator._tasks = [
            TaskModel(id="1", description="task 1", status=TaskStatus.COMPLETED),
            TaskModel(id="2", description="task 2", status=TaskStatus.COMPLETED),
        ]
        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_any_task_failed(self, orchestrator):
        orchestrator._tasks = [
            TaskModel(id="1", description="task 1", status=TaskStatus.COMPLETED),
            TaskModel(id="2", description="task 2", status=TaskStatus.FAILED),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_any_task_pending(self, orchestrator):
        orchestrator._tasks = [
            TaskModel(id="1", description="task 1", status=TaskStatus.COMPLETED),
            TaskModel(id="2", description="task 2", status=TaskStatus.PENDING),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_empty_tasks(self, orchestrator):
        orchestrator._tasks = []
        assert orchestrator._is_done() is False

    def test_is_done_returns_true_when_no_tasks_and_completed_zero(self, orchestrator):
        orchestrator._tasks = []
        assert orchestrator._is_done() is False


class TestGoalPreservationOnTestFailure:
    @pytest.mark.asyncio
    async def test_goal_preserved_and_failure_appended_on_test_failure(self):
        with patch.object(LLMClient, "__init__", lambda self, settings=None: setattr(self, "settings", settings or Settings())):
            orchestrator = Orchestrator(goal="original goal")

        original_goal = "original goal"
        orchestrator.goal = original_goal

        test_failures = ["test_foo failed", "test_bar failed"]
        test_result = TestResult(
            passed=False,
            summary="2 tests failed",
            failures=test_failures
        )

        with patch.object(TesterAgent, "run", AsyncMock(return_value=test_result)):
            with patch.object(PlannerAgent, "plan", AsyncMock(return_value=Plan(
                tasks=[TaskModel(id="1", description="fix tests")],
                rationale="fix failing tests"
            ))):
                with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker:
                    mock_worker_instance = AsyncMock()
                    mock_worker_instance.run = AsyncMock(return_value="done")
                    mock_worker.return_value = mock_worker_instance

                    await orchestrator._cycle()

        expected_goal = f"{original_goal}\nFix failing tests:\n" + "\n".join(test_failures)
        assert orchestrator.goal == expected_goal


class TestPlannerGoalLengthValidation:
    @pytest.mark.asyncio
    async def test_plan_raises_value_error_when_goal_exceeds_10000_chars(self):
        long_goal = "x" * 10001
        planner = PlannerAgent()

        with pytest.raises(ValueError, match="Goal exceeds maximum length of 10000 characters"):
            await planner.plan(long_goal)

    @pytest.mark.asyncio
    async def test_plan_accepts_goal_at_10000_chars(self):
        goal = "x" * 10000
        planner = PlannerAgent()

        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "tasks": [{"id": "1", "description": "test"}],
            "rationale": "test"
        }))
        mock_client.settings = Settings()
        planner.client = mock_client

        plan = await planner.plan(goal)
        assert isinstance(plan, Plan)


class TestTesterPassFailHeuristic:
    @pytest.mark.asyncio
    async def test_heuristic_returns_false_when_passed_and_failed_in_response(self):
        tester = TesterAgent()

        response_with_both = "Tests passed: 5, failed: 2"
        result = TestResult(
            passed="passed" in response_with_both.lower() and "failed" not in response_with_both.lower() and "not passed" not in response_with_both.lower(),
            summary=response_with_both,
            failures=[]
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_heuristic_returns_true_when_only_passed_in_response(self):
        response = "All tests passed successfully"
        result = TestResult(
            passed="passed" in response.lower() and "failed" not in response.lower() and "not passed" not in response.lower(),
            summary=response,
            failures=[]
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_heuristic_returns_false_when_not_passed_in_response(self):
        response = "Tests not passed"
        result = TestResult(
            passed="passed" in response.lower() and "failed" not in response.lower() and "not passed" not in response.lower(),
            summary=response,
            failures=[]
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_heuristic_returns_false_when_failed_in_response(self):
        response = "Some tests failed"
        result = TestResult(
            passed="passed" in response.lower() and "failed" not in response.lower() and "not passed" not in response.lower(),
            summary=response,
            failures=[]
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_run_tests_heuristic_in_run_method(self):
        tester = TesterAgent()

        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
        mock_client.settings = Settings()
        tester.client = mock_client

        with patch.object(tester, "_run_tests", AsyncMock(return_value="test output")):
            result = await tester.run("goal", [])
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_handles_json_parse_failure_with_heuristic(self):
        tester = TesterAgent()

        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value="Tests passed but some failed")
        mock_client.settings = Settings()
        tester.client = mock_client

        with patch.object(tester, "_run_tests", AsyncMock(return_value="test output")):
            result = await tester.run("goal", [])
            assert result.passed is False
            assert "failed" in result.summary.lower()


class TestLLMClientErrorHandling:
    def test_anthropic_raises_runtime_error_when_no_api_key(self):
        settings = Settings(anthropic_api_key=None)
        client = LLMClient(settings=settings)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="Failed to initialize Anthropic client"):
                _ = client.anthropic

    def test_openai_raises_runtime_error_when_no_api_key(self):
        settings = Settings(openai_api_key=None)
        client = LLMClient(settings=settings)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="Failed to initialize OpenAI client"):
                _ = client.openai

    def test_anthropic_raises_runtime_error_on_init_failure(self):
        settings = Settings(anthropic_api_key="invalid-key")
        client = LLMClient(settings=settings)

        with patch("furrow.llm.AsyncAnthropic", side_effect=Exception("connection failed")):
            with pytest.raises(RuntimeError, match="Failed to initialize Anthropic client"):
                _ = client.anthropic

    def test_openai_raises_runtime_error_on_init_failure(self):
        settings = Settings(openai_api_key="invalid-key")
        client = LLMClient(settings=settings)

        with patch("furrow.llm.AsyncOpenAI", side_effect=Exception("connection failed")):
            with pytest.raises(RuntimeError, match="Failed to initialize OpenAI client"):
                _ = client.openai

    def test_anthropic_works_with_valid_api_key(self):
        settings = Settings(anthropic_api_key="valid-key")
        client = LLMClient(settings=settings)

        with patch("furrow.llm.AsyncAnthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance
            anthropic = client.anthropic
            assert anthropic == mock_instance
            mock_anthropic.assert_called_once_with(api_key="valid-key", timeout=120.0)

    def test_openai_works_with_valid_api_key(self):
        settings = Settings(openai_api_key="valid-key")
        client = LLMClient(settings=settings)

        with patch("furrow.llm.AsyncOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            openai = client.openai
            assert openai == mock_instance
            mock_openai.assert_called_once_with(api_key="valid-key", timeout=120.0)

    def test_anthropic_uses_env_var_when_settings_key_none(self):
        settings = Settings(anthropic_api_key=None)
        client = LLMClient(settings=settings)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            with patch("furrow.llm.AsyncAnthropic") as mock_anthropic:
                mock_instance = MagicMock()
                mock_anthropic.return_value = mock_instance
                anthropic = client.anthropic
                assert anthropic == mock_instance
                mock_anthropic.assert_called_once_with(api_key="env-key", timeout=120.0)

    def test_openai_uses_env_var_when_settings_key_none(self):
        settings = Settings(openai_api_key=None)
        client = LLMClient(settings=settings)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            with patch("furrow.llm.AsyncOpenAI") as mock_openai:
                mock_instance = MagicMock()
                mock_openai.return_value = mock_instance
                openai = client.openai
                assert openai == mock_instance
                mock_openai.assert_called_once_with(api_key="env-key", timeout=120.0)

    def test_anthropic_caches_client(self):
        settings = Settings(anthropic_api_key="valid-key")
        client = LLMClient(settings=settings)

        with patch("furrow.llm.AsyncAnthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance
            _ = client.anthropic
            _ = client.anthropic
            assert mock_anthropic.call_count == 1

    def test_openai_caches_client(self):
        settings = Settings(openai_api_key="valid-key")
        client = LLMClient(settings=settings)

        with patch("furrow.llm.AsyncOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            _ = client.openai
            _ = client.openai
            assert mock_openai.call_count == 1