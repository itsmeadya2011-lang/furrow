import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestConfigModels:
    def test_plan_parse(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert p.tasks[0].description == "do thing"

    def test_test_result(self):
        t = TestResult(passed=True, summary="ok", failures=[])
        assert t.passed is True

    def test_task_model_defaults(self):
        t = TaskModel(id="1", description="x")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_valid_json(self):
        client = AsyncMock(spec=LLMClient)
        client.settings.planner_model = "test-model"
        client.complete = AsyncMock(return_value='{"tasks": [{"id":"1","description":"test"}], "rationale": "ok"}')
        planner = PlannerAgent(client=client)
        plan = await planner.plan("do test")
        assert len(plan.tasks) == 1
        assert plan.rationale == "ok"

    @pytest.mark.asyncio
    async def test_plan_json_in_code_block(self):
        client = AsyncMock(spec=LLMClient)
        client.settings.planner_model = "test-model"
        client.complete = AsyncMock(return_value='```json\n{"tasks": [], "rationale": "none"}\n```')
        planner = PlannerAgent(client=client)
        plan = await planner.plan("nothing")
        assert plan.tasks == []

    @pytest.mark.asyncio
    async def test_plan_fallback_on_garbage(self):
        client = AsyncMock(spec=LLMClient)
        client.settings.planner_model = "test-model"
        client.complete = AsyncMock(return_value="This is not JSON at all.")
        planner = PlannerAgent(client=client)
        plan = await planner.plan("bad input")
        assert plan.tasks == []
        assert "Failed to parse" in plan.rationale


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_no_files(self):
        client = AsyncMock(spec=LLMClient)
        client.settings.worker_model = "test-model"
        client.complete = AsyncMock(return_value="summary here")
        client.list_files = MagicMock(return_value=[])
        worker = WorkerAgent(task=TaskModel(id="1", description="add feature"), client=client)
        result = await worker.run()
        assert "Wrote 0 file(s)" in result

    @pytest.mark.asyncio
    async def test_worker_writes_files(self):
        client = AsyncMock(spec=LLMClient)
        client.settings.worker_model = "test-model"
        client.complete = AsyncMock(
            return_value='<write path="src/main.py">\n<![CDATA[print("hi")]]>\n</write>\n\nDone.'
        )
        client.list_files = MagicMock(return_value=[])
        client.write_file = AsyncMock()
        worker = WorkerAgent(task=TaskModel(id="1", description="add feature"), client=client)
        result = await worker.run()
        client.write_file.assert_called_once_with("src/main.py", 'print("hi")')
        assert "src/main.py" in result

    @pytest.mark.asyncio
    async def test_worker_reads_existing_files(self):
        client = AsyncMock(spec=LLMClient)
        client.settings.worker_model = "test-model"
        client.complete = AsyncMock(return_value="no writes")
        client.read_file = AsyncMock(return_value="existing content")
        client.list_files = MagicMock(return_value=[])
        worker = WorkerAgent(
            task=TaskModel(id="1", description="edit", files=["src/main.py"]),
            client=client,
        )
        await worker.run()
        client.read_file.assert_called_once_with("src/main.py")


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_single_cycle_completion(self):
        client = AsyncMock(spec=LLMClient)
        client.settings.planner_model = "test-model"
        client.settings.worker_model = "test-model"
        client.settings.tester_model = "test-model"
        client.settings.max_cycles = 1
        client.settings.max_parallel_tasks = 2

        plan_json = '{"tasks": [{"id":"1","description":"t","files":[]}], "rationale": "r"}'
        client.complete = AsyncMock(side_effect=[plan_json, "Done.", '{"passed": true, "summary": "ok", "failures": []}'])

        orchestrator = Orchestrator(goal="test", client=client, settings=client.settings)
        await orchestrator.run()
        assert orchestrator.cycles == 1

    @pytest.mark.asyncio
    async def test_is_done_true_when_all_completed(self):
        client = AsyncMock(spec=LLMClient)
        client.settings.max_cycles = 0
        client.settings.max_parallel_tasks = 2
        plan_json = '{"tasks": [{"id":"1","description":"t","files":[]}], "rationale": "r"}'
        client.complete = AsyncMock(side_effect=[plan_json, "Done.", '{"passed": true, "summary": "ok", "failures": []}'])

        orchestrator = Orchestrator(goal="test", client=client, settings=client.settings)
        await orchestrator.run()
        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_max_cycles_enforced(self):
        client = AsyncMock(spec=LLMClient)
        client.settings.max_cycles = 2
        client.settings.max_parallel_tasks = 2
        client.settings.planner_model = "test-model"
        client.settings.worker_model = "test-model"
        client.settings.tester_model = "test-model"

        plan_json = '{"tasks": [{"id":"1","description":"t","files":[]}], "rationale": "r"}'
        fail_json = '{"passed": false, "summary": "fail", "failures": ["err"]}'
        side_effects = [plan_json, "Done.", fail_json, plan_json, "Done.", fail_json]
        client.complete = AsyncMock(side_effect=side_effects)

        orchestrator = Orchestrator(goal="test", client=client, settings=client.settings)
        await orchestrator.run()
        assert orchestrator.cycles == 2


class TestLLMClient:
    def test_unsupported_provider_raises(self):
        client = LLMClient.__new__(LLMClient)
        client.settings = type("S", (), {"provider": "unknown", "model": "x"})()
        with pytest.raises(ValueError, match="Unsupported provider"):
            import asyncio
            asyncio.run(client.complete("test"))
