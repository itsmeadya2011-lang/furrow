import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestOrchestrator:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings.planner_model = "test-model"
        client.settings.worker_model = "test-model"
        client.settings.tester_model = "test-model"
        return client

    @pytest.mark.asyncio
    async def test_is_done_with_no_tasks(self, mock_client):
        orchestrator = Orchestrator(goal="test", client=mock_client)
        # Before any plan, not done
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_completed_tasks(self, mock_client):
        orchestrator = Orchestrator(goal="test", client=mock_client)
        orchestrator.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="ok",
        )
        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_failed_task(self, mock_client):
        orchestrator = Orchestrator(goal="test", client=mock_client)
        orchestrator.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="ok",
        )
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_partial_completion(self, mock_client):
        orchestrator = Orchestrator(goal="test", client=mock_client)
        orchestrator.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="pending"),
            ],
            rationale="ok",
        )
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_run_honors_max_cycles(self, mock_client):
        orchestrator = Orchestrator(goal="test", client=mock_client, max_cycles=1)
        # Patch planner to return a plan with a task so we get past planning
        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = Plan(
                tasks=[TaskModel(id="1", description="do it")],
                rationale="ok",
            )
            # Patch worker to avoid real LLM calls
            mock_worker_run = AsyncMock(return_value="done")
            with patch("furrow.core.orchestrator.WorkerAgent", return_value=MagicMock(run=mock_worker_run)):
                # Patch tester to pass so we don't loop forever on failures
                with patch(
                    "furrow.core.orchestrator.TesterAgent",
                    return_value=MagicMock(
                        run=AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))
                    ),
                ):
                    await orchestrator.run()
        assert orchestrator.cycles == 1
        mock_plan.assert_called_once()
        mock_worker_run.assert_called_once()


class TestLLMClient:
    def test_read_file(self, tmp_path):
        async def _test():
            client = LLMClient()
            test_file = tmp_path / "test.txt"
            test_file.write_text("hello")
            result = await client.read_file(str(test_file))
            assert result == "hello"

        asyncio.run(_test())

    def test_write_file(self, tmp_path):
        async def _test():
            client = LLMClient()
            test_file = tmp_path / "sub" / "test.txt"
            await client.write_file(str(test_file), "hello")
            assert test_file.read_text() == "hello"
            assert test_file.exists()

        asyncio.run(_test())

    def test_list_files(self, tmp_path):
        async def _test():
            client = LLMClient()
            (tmp_path / "a.txt").write_text("a")
            (tmp_path / "b.txt").write_text("b")
            (tmp_path / "sub").mkdir()
            (tmp_path / "sub" / "c.txt").write_text("c")
            files = client.list_files(str(tmp_path))
            assert set(files) == {"a.txt", "b.txt", "sub/c.txt"}

        asyncio.run(_test())


class TestPlannerPrompt:
    def test_planner_prompt_includes_context(self):
        from furrow.agents.prompts import PLANNER_PROMPT

        prompt = PLANNER_PROMPT.format(context="\nContext:\n- Original goal: test\n- Cycle: 1\n")
        assert "Original goal: test" in prompt
        assert "Cycle: 1" in prompt


class TestWorkerPrompt:
    def test_worker_prompt_includes_goal_and_cycle(self):
        from furrow.agents.prompts import WORKER_PROMPT

        prompt = WORKER_PROMPT.format(goal="build auth", cycle=2)
        assert "build auth" in prompt
        assert "Cycle: 2" in prompt


class TestTesterPrompt:
    def test_tester_prompt_includes_tasks(self):
        from furrow.agents.prompts import TESTER_PROMPT

        tasks = "- 1: do thing\n- 2: do other"
        prompt = TESTER_PROMPT.format(goal="test goal", tasks=tasks)
        assert "test goal" in prompt
        assert "do thing" in prompt
