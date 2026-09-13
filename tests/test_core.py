import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_task():
    return TaskModel(id="1", description="do thing")


@pytest.fixture
def sample_plan():
    return Plan(
        tasks=[
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2", files=["src/a.py"]),
        ],
        rationale="test rationale",
    )


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value="ok")
    client.settings = Settings()
    return client


# ── 1. Plan/Tasks serialization ───────────────────────────────────────────


class TestPlanSerialization:
    def test_model_dump(self, sample_plan):
        d = sample_plan.model_dump()
        assert d["rationale"] == "test rationale"
        assert len(d["tasks"]) == 2
        assert d["tasks"][0]["id"] == "1"
        assert d["tasks"][0]["status"] == "pending"
        assert d["tasks"][0]["result"] is None

    def test_model_dump_json(self, sample_plan):
        s = sample_plan.model_dump_json()
        d = json.loads(s)
        assert d["rationale"] == "test rationale"
        assert len(d["tasks"]) == 2
        assert d["tasks"][1]["files"] == ["src/a.py"]

    def test_plan_required_fields_missing_raises(self):
        with pytest.raises(Exception):
            Plan()

    def test_task_required_fields_missing_raises(self):
        with pytest.raises(Exception):
            TaskModel(id="1")

    def test_invalid_data_rejected(self):
        with pytest.raises(Exception):
            Plan(tasks="not_a_list", rationale="ok")

        with pytest.raises(Exception):
            TaskModel(id=None, description="ok")


# ── 2. TaskModel state transitions ────────────────────────────────────────


class TestTaskModelStateTransitions:
    def test_pending_to_completed(self):
        task = TaskModel(id="1", description="test")
        assert task.status == "pending"
        task.status = "completed"
        assert task.status == "completed"

    def test_pending_to_failed(self):
        task = TaskModel(id="1", description="test")
        assert task.status == "pending"
        task.status = "failed"
        assert task.status == "failed"

    def test_task_with_result(self):
        task = TaskModel(id="1", description="test", result="done")
        assert task.result == "done"
        assert task.status == "pending"


# ── 3. Provider enum ──────────────────────────────────────────────────────


class TestProviderEnum:
    def test_all_values(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"

    def test_provider_values_set(self):
        values = {p.value for p in Provider}
        assert values == {"anthropic", "openai", "ollama"}

    def test_invalid_value_rejected(self):
        with pytest.raises(Exception):
            Provider("invalid_provider")

    def test_provider_from_string(self):
        p = Provider("anthropic")
        assert p == Provider.ANTHROPIC


# ── 4. Settings ────────────────────────────────────────────────────────────


class TestSettings:
    def test_default_provider(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC

    def test_default_model(self):
        s = Settings()
        assert s.model == "claude-sonnet-4-20250514"

    def test_planner_model_default(self):
        s = Settings()
        assert s.planner_model == "claude-3-5-haiku-20241022"

    def test_worker_model_default(self):
        s = Settings()
        assert s.worker_model == "claude-3-5-sonnet-20241022"

    def test_tester_model_default(self):
        s = Settings()
        assert s.tester_model == "claude-3-5-sonnet-20241022"

    def test_default_anthropic_api_key_none(self):
        s = Settings()
        assert s.anthropic_api_key is None

    def test_default_openai_api_key_none(self):
        s = Settings()
        assert s.openai_api_key is None

    def test_default_ollama_base_url(self):
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_default_max_parallel_tasks(self):
        s = Settings()
        assert s.max_parallel_tasks == 5

    def test_default_max_cycles(self):
        s = Settings()
        assert s.max_cycles == 0

    def test_default_workspace_is_path(self):
        s = Settings()
        assert isinstance(s.workspace, Path)

    def test_default_log_level(self):
        s = Settings()
        assert s.log_level == "INFO"

    def test_env_var_override_provider(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        s = Settings()
        assert s.provider == Provider.OPENAI

    def test_env_var_override_model(self, monkeypatch):
        monkeypatch.setenv("FURROW_MODEL", "gpt-4")
        s = Settings()
        assert s.model == "gpt-4"

    def test_env_var_override_workspace(self, monkeypatch):
        monkeypatch.setenv("FURROW_WORKSPACE", "/tmp/test_ws")
        s = Settings()
        assert str(s.workspace) == "/tmp/test_ws"


# ── 5. LLMClient instantiation ────────────────────────────────────────────


class TestLLMClient:
    def test_instantiation_without_api_keys(self):
        client = LLMClient()
        assert client.settings is not None

    def test_anthropic_property_raises_without_key(self):
        client = LLMClient()
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            _ = client.anthropic

    def test_openai_property_raises_without_key(self):
        client = LLMClient()
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _ = client.openai

    def test_anthropic_property_with_key(self):
        client = LLMClient(settings=Settings(anthropic_api_key="test-key"))
        with patch("furrow.llm.AsyncAnthropic") as mock_cls:
            _ = client.anthropic
            mock_cls.assert_called_once_with(api_key="test-key")

    def test_openai_property_with_key(self):
        client = LLMClient(settings=Settings(openai_api_key="test-key"))
        with patch("furrow.llm.AsyncOpenAI") as mock_cls:
            _ = client.openai
            mock_cls.assert_called_once_with(api_key="test-key")

    def test_provider_selection_anthropic(self):
        client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
        assert client.settings.provider == Provider.ANTHROPIC

    def test_provider_selection_openai(self):
        client = LLMClient(settings=Settings(provider=Provider.OPENAI))
        assert client.settings.provider == Provider.OPENAI

    def test_provider_selection_ollama(self):
        client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
        assert client.settings.provider == Provider.OLLAMA

    def test_complete_raises_for_unsupported_provider(self):
        client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
        with pytest.raises(ValueError, match="Unsupported provider"):
            asyncio.run(client.complete("prompt"))

    def test_complete_raises_without_api_key(self):
        client = LLMClient()
        with pytest.raises(ValueError):
            asyncio.run(client.complete("prompt"))


# ── 6. PlannerAgent ────────────────────────────────────────────────────────


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_returns_valid_plan(self, mock_llm_client):
        mock_llm_client.complete.return_value = json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "do thing", "files": [], "dependencies": []}
                ],
                "rationale": "test",
            }
        )
        agent = PlannerAgent(client=mock_llm_client)
        plan = await agent.plan("test goal")
        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do thing"
        assert plan.rationale == "test"

    @pytest.mark.asyncio
    async def test_plan_calls_complete_with_goal(self, mock_llm_client):
        mock_llm_client.complete.return_value = json.dumps(
            {"tasks": [], "rationale": "empty"}
        )
        agent = PlannerAgent(client=mock_llm_client)
        await agent.plan("my goal")
        mock_llm_client.complete.assert_called_once()
        call_args = mock_llm_client.complete.call_args
        assert "my goal" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_plan_raises_on_bad_json(self, mock_llm_client):
        mock_llm_client.complete.return_value = "not json"
        agent = PlannerAgent(client=mock_llm_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("test goal")

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_plan_schema(self, mock_llm_client):
        mock_llm_client.complete.return_value = json.dumps(
            {"tasks": "bad", "rationale": "ok"}
        )
        agent = PlannerAgent(client=mock_llm_client)
        with pytest.raises(Exception):
            await agent.plan("test goal")


# ── 7. WorkerAgent ────────────────────────────────────────────────────────


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_returns_dict(self, mock_llm_client, sample_task):
        mock_llm_client.complete.return_value = "implemented feature"
        agent = WorkerAgent(task=sample_task, client=mock_llm_client)
        result = await agent.run()
        assert isinstance(result, dict)
        assert "summary" in result
        assert "files_written" in result
        assert "issues" in result
        assert result["summary"] != ""

    @pytest.mark.asyncio
    async def test_run_calls_complete_with_task_description(self, mock_llm_client, sample_task):
        mock_llm_client.complete.return_value = "done"
        agent = WorkerAgent(task=sample_task, client=mock_llm_client)
        await agent.run()
        mock_llm_client.complete.assert_called_once()
        call_args = mock_llm_client.complete.call_args
        assert sample_task.description in call_args[0][0]

    @pytest.mark.asyncio
    async def test_run_with_files(self, mock_llm_client):
        task = TaskModel(
            id="2", description="edit file", files=["src/main.py", "src/util.py"]
        )
        mock_llm_client.complete.return_value = "done"
        agent = WorkerAgent(task=task, client=mock_llm_client)
        result = await agent.run()
        assert isinstance(result, dict)
        assert "files_written" in result
        assert "issues" in result
        call_args = mock_llm_client.complete.call_args
        assert "src/main.py" in call_args[0][0]
        assert "src/util.py" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_run_returns_issues_on_exception(self, mock_llm_client, sample_task):
        mock_llm_client.complete = AsyncMock(side_effect=RuntimeError("LLM error"))
        agent = WorkerAgent(task=sample_task, client=mock_llm_client)
        result = await agent.run()
        assert len(result["issues"]) > 0
        assert "LLM error" in result["issues"]

    @pytest.mark.asyncio
    async def test_run_returns_empty_files_written_when_no_code(self, mock_llm_client, sample_task):
        mock_llm_client.complete.return_value = "no code block here"
        agent = WorkerAgent(task=sample_task, client=mock_llm_client)
        result = await agent.run()
        assert isinstance(result["files_written"], list)


# ── 8. TesterAgent ────────────────────────────────────────────────────────


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_passing_results(self, mock_llm_client, sample_task):
        mock_llm_client.complete.return_value = json.dumps(
            {"passed": True, "summary": "all passed", "failures": []}
        )
        agent = TesterAgent(client=mock_llm_client)
        result = await agent.run("test goal", [sample_task])
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "all passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_failing_results(self, mock_llm_client, sample_task):
        mock_llm_client.complete.return_value = json.dumps(
            {
                "passed": False,
                "summary": "1 test failed",
                "failures": ["test_x failed"],
            }
        )
        agent = TesterAgent(client=mock_llm_client)
        result = await agent.run("test goal", [sample_task])
        assert isinstance(result, TestResult)
        assert result.passed is False
        assert result.summary == "1 test failed"
        assert result.failures == ["test_x failed"]

    @pytest.mark.asyncio
    async def test_run_returns_false_on_test_runner_exception(self, mock_llm_client, sample_task):
        agent = TesterAgent(client=mock_llm_client)
        with patch.object(
            agent, "_run_tests", new=AsyncMock(side_effect=RuntimeError("no tests"))
        ):
            result = await agent.run("test goal", [sample_task])
        assert result.passed is False
        assert "no tests" in result.summary

    @pytest.mark.asyncio
    async def test_run_parses_response_when_json_missing_key(self, mock_llm_client, sample_task):
        mock_llm_client.complete.return_value = "response text"
        agent = TesterAgent(client=mock_llm_client)
        with patch.object(agent, "_run_tests", new=AsyncMock(return_value="output")):
            result = await agent.run("test goal", [sample_task])
        assert isinstance(result, TestResult)

    @pytest.mark.asyncio
    async def test_run_calls_complete_with_goal_and_test_output(self, mock_llm_client, sample_task):
        mock_llm_client.complete.return_value = json.dumps(
            {"passed": True, "summary": "ok", "failures": []}
        )
        agent = TesterAgent(client=mock_llm_client)
        with patch.object(agent, "_run_tests", new=AsyncMock(return_value="test output")):
            await agent.run("my goal", [sample_task])
        mock_llm_client.complete.assert_called_once()
        call_args = mock_llm_client.complete.call_args
        assert "my goal" in call_args[0][0]
        assert "test output" in call_args[0][0]


# ── 9. Orchestrator ───────────────────────────────────────────────────────


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_init_with_client(self):
        client = MagicMock(spec=LLMClient)
        client.complete = AsyncMock(return_value="ok")
        client.settings = Settings()
        orch = Orchestrator(goal="test goal", client=client)
        assert orch.goal == "test goal"
        assert orch.planner is not None
        assert orch.cycles == 0

    def test_is_done_no_tasks(self):
        orch = Orchestrator(goal="test")
        assert orch._is_done() is True

    def test_is_done_all_completed(self):
        orch = Orchestrator(goal="test")
        with patch.object(orch, "_get_tasks", return_value=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]):
            tasks = orch._get_tasks()
            for t in tasks:
                t.status = "completed"
            assert orch._is_done() is True

    def test_is_done_with_failed_task(self):
        orch = Orchestrator(goal="test")
        with patch.object(orch, "_get_tasks", return_value=[TaskModel(id="1", description="a")]):
            tasks = orch._get_tasks()
            tasks[0].status = "failed"
            assert orch._is_done() is False

    def test_is_done_partial_completion(self):
        orch = Orchestrator(goal="test")
        with patch.object(orch, "_get_tasks", return_value=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]):
            tasks = orch._get_tasks()
            tasks[0].status = "completed"
            tasks[1].status = "pending"
            assert orch._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_with_mixed_status(self):
        orch = Orchestrator(goal="test")
        with patch.object(orch, "_get_tasks", return_value=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]):
            tasks = orch._get_tasks()
            tasks[0].status = "completed"
            tasks[1].status = "failed"
            assert orch._is_done() is False

    @pytest.mark.asyncio
    async def test_cycle_flow(self, mock_llm_client):
        orch = Orchestrator(goal="test goal", client=mock_llm_client)
        plan = Plan(
            tasks=[TaskModel(id="1", description="task 1")],
            rationale="test",
        )
        mock_plan = AsyncMock(return_value=plan)
        with patch.object(orch.planner, "plan", mock_plan):
            mock_worker_run = AsyncMock(return_value="done")
            with patch.object(WorkerAgent, "run", mock_worker_run):
                with patch.object(
                    TesterAgent, "run", new=AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))
                ):
                    with patch("furrow.core.orchestrator.console"):
                        await orch._cycle()
                        mock_plan.assert_called_once_with("test goal")
                        mock_worker_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_with_no_tasks(self, mock_llm_client):
        orch = Orchestrator(goal="test goal", client=mock_llm_client)
        with patch.object(
            orch.planner, "plan", new=AsyncMock(return_value=Plan(tasks=[], rationale="none"))
        ):
            with patch("furrow.core.orchestrator.console") as mock_console:
                await orch._cycle()
                mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_run_cycles_and_stops_when_done(self, mock_llm_client):
        mock_llm_client.complete.return_value = json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "task 1", "files": [], "dependencies": []}
                ],
                "rationale": "test",
            }
        )
        orch = Orchestrator(goal="test goal", client=mock_llm_client)
        plan_with_task = Plan(
            tasks=[TaskModel(id="1", description="task 1", status="completed", result="done")],
            rationale="test",
        )
        empty_plan = Plan(tasks=[], rationale="done")
        with patch.object(orch.planner, "plan", new=AsyncMock(side_effect=[plan_with_task, empty_plan])):
            with patch.object(orch, "_cycle", new=AsyncMock()):
                with patch.object(orch, "_is_done", return_value=True):
                    await orch.run()
                    assert orch.cycles == 1


# ── 10. Concurrency ───────────────────────────────────────────────────────


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_asyncio_gather_with_multiple_workers(self, mock_llm_client):
        tasks_data = [
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
            TaskModel(id="3", description="task 3"),
        ]
        results = []

        async def slow_run(task, delay):
            await asyncio.sleep(delay)
            return f"result for {task.id}"

        coros = [slow_run(t, 0.01 * (i + 1)) for i, t in enumerate(tasks_data)]
        results = await asyncio.gather(*coros, return_exceptions=True)

        assert len(results) == 3
        assert "result for 1" in results
        assert "result for 2" in results
        assert "result for 3" in results

    @pytest.mark.asyncio
    async def test_parallel_execution_faster_than_sequential(self):
        async def slow_task(n):
            await asyncio.sleep(0.05)
            return n

        tasks = [slow_task(i) for i in range(4)]

        start = time.perf_counter()
        parallel_results = await asyncio.gather(*tasks)
        parallel_time = time.perf_counter() - start

        start = time.perf_counter()
        sequential_results = []
        for t in tasks:
            sequential_results.append(await t)
        sequential_time = time.perf_counter() - start

        assert parallel_results == sequential_results
        assert parallel_time < sequential_time

    @pytest.mark.asyncio
    async def test_worker_run_in_parallel(self, mock_llm_client):
        task1 = TaskModel(id="1", description="task 1")
        task2 = TaskModel(id="2", description="task 2")

        def make_client(result):
            c = MagicMock(spec=LLMClient)
            c.complete = AsyncMock(return_value=result)
            c.settings = Settings()
            return c

        worker1 = WorkerAgent(task=task1, client=make_client("result1"))
        worker2 = WorkerAgent(task=task2, client=make_client("result2"))

        results = await asyncio.gather(
            worker1.run(), worker2.run(), return_exceptions=True
        )
        assert isinstance(results[0], dict)
        assert "summary" in results[0]
        assert "result1" in results[0]["summary"]
        assert isinstance(results[1], dict)
        assert "summary" in results[1]
        assert "result2" in results[1]["summary"]

    @pytest.mark.asyncio
    async def test_gather_handles_exceptions(self):
        async def failing_task():
            raise ValueError("error")

        async def success_task():
            return "ok"

        results = await asyncio.gather(
            failing_task(), success_task(), return_exceptions=True
        )
        assert isinstance(results[0], ValueError)
        assert str(results[0]) == "error"
        assert results[1] == "ok"

    @pytest.mark.asyncio
    async def test_orchestrator_cycle_uses_asyncio_gather(self, mock_llm_client):
        orch = Orchestrator(goal="test", client=mock_llm_client)
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="t1"),
                TaskModel(id="2", description="t2"),
            ],
            rationale="test",
        )
        with patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)):
            with patch("asyncio.gather", new=AsyncMock(return_value=["r1", "r2"])) as mock_gather:
                with patch("furrow.core.orchestrator.console"):
                    with patch.object(
                        TesterAgent, "run", new=AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))
                    ):
                        await orch._cycle()
                        mock_gather.assert_called_once()
                        call_kwargs = mock_gather.call_args
                        assert len(call_kwargs[0]) == 2

    @pytest.mark.asyncio
    async def test_concurrent_workers_get_different_results(self, mock_llm_client):
        tasks = [TaskModel(id=str(i), description=f"task {i}") for i in range(5)]

        def make_client(i):
            c = MagicMock(spec=LLMClient)
            c.complete = AsyncMock(return_value=f"result_{i}")
            c.settings = Settings()
            return c

        workers = [WorkerAgent(task=t, client=make_client(i)) for i, t in enumerate(tasks)]
        results = await asyncio.gather(*(w.run() for w in workers), return_exceptions=True)

        for i, r in enumerate(results):
            assert isinstance(r, dict)
            assert "summary" in r
            assert f"result_{i}" in r["summary"]
