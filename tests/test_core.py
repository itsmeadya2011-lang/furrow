from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import tenacity

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

TestResult.__test__ = False
TesterAgent.__test__ = False


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class TestOrchestrator:
    def test_is_done_true_when_all_tasks_completed(self, mock_client):
        orch = Orchestrator(goal="g", client=mock_client)
        orch._current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="ok",
        )
        assert orch._is_done() is True

    def test_is_done_false_when_any_task_failed(self, mock_client):
        orch = Orchestrator(goal="g", client=mock_client)
        orch._current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="ok",
        )
        assert orch._is_done() is False

    def test_is_done_false_when_no_plan(self, mock_client):
        orch = Orchestrator(goal="g", client=mock_client)
        assert orch._is_done() is False

    def test_get_tasks_returns_current_plan_tasks(self, mock_client):
        orch = Orchestrator(goal="g", client=mock_client)
        tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
        orch._current_plan = Plan(tasks=tasks, rationale="ok")
        assert orch._get_tasks() == tasks

    def test_get_tasks_returns_empty_when_no_plan(self, mock_client):
        orch = Orchestrator(goal="g", client=mock_client)
        assert orch._get_tasks() == []

    def test_constructed_with_custom_llm_client(self, mock_client):
        orch = Orchestrator(goal="g", client=mock_client)
        assert orch.client is mock_client
        assert orch.planner.client is mock_client

    async def test_max_cycles_stops_loop(self, mock_client, monkeypatch):
        mock_client.settings.max_cycles = 2
        orch = Orchestrator(goal="g", client=mock_client)
        cycle_mock = AsyncMock(return_value=False)
        monkeypatch.setattr(Orchestrator, "_cycle", cycle_mock)
        await orch.run()
        assert orch.cycles == 2
        assert cycle_mock.call_count == 1


class TestWorkerAgent:
    async def test_writes_files_from_file_markers(self, mock_client, tmp_workspace):
        task = TaskModel(id="1", description="create foo", files=[])
        worker = WorkerAgent(task=task, client=mock_client)
        mock_client.responses = [
            'FILE: src/foo.py\n```python\nprint("hello")\n```\n\nSummary: wrote foo module'
        ]
        result = await worker.run()

        written = (tmp_workspace / "src" / "foo.py").read_text()
        assert written == 'print("hello")'
        assert "Wrote files: src/foo.py" in result
        assert "Summary: wrote foo module" in result

    async def test_returns_raw_text_when_no_file_blocks(self, mock_client):
        task = TaskModel(id="1", description="do thing", files=[])
        worker = WorkerAgent(task=task, client=mock_client)
        mock_client.responses = ["Just plain text response."]
        result = await worker.run()
        assert result == "Just plain text response."

    async def test_reads_existing_files_for_context(self, mock_client, tmp_workspace):
        (tmp_workspace / "existing.txt").write_text("original content")
        task = TaskModel(id="1", description="update", files=["existing.txt"])
        worker = WorkerAgent(task=task, client=mock_client)
        mock_client.responses = ["FILE: existing.txt\n```python\nupdated\n```\n"]
        await worker.run()

        call_prompt = mock_client.complete_calls[-1]["prompt"]
        assert "original content" in call_prompt
        assert (tmp_workspace / "existing.txt").read_text() == "updated"


class TestPlannerAgent:
    async def test_parses_valid_json_plan(self, mock_client):
        planner = PlannerAgent(client=mock_client)
        mock_client.responses = [
            '{"tasks":[{"id":"1","description":"do thing","files":[]}],"rationale":"ok"}'
        ]
        plan = await planner.plan("do thing")
        assert isinstance(plan, Plan)
        assert plan.tasks[0].description == "do thing"
        assert plan.rationale == "ok"

    async def test_raises_value_error_on_invalid_json(self, mock_client):
        planner = PlannerAgent(client=mock_client)
        mock_client.responses = ["not valid json {{{"]
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("do thing")


class TestTesterAgent:
    async def test_parses_valid_json_test_results(self, mock_client):
        tester = TesterAgent(client=mock_client)
        tester._run_tests = AsyncMock(return_value="tests ran ok")
        mock_client.responses = ['{"passed":true,"summary":"All passed","failures":[]}']
        result = await tester.run("goal", [])
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All passed"

    async def test_handles_missing_test_runner_gracefully(self, mock_client):
        tester = TesterAgent(client=mock_client)
        tester._run_tests = AsyncMock(return_value="No test runner found.")
        mock_client.responses = ['{"passed":false,"summary":"no runner","failures":[]}']
        result = await tester.run("goal", [])
        assert isinstance(result, TestResult)
        assert result.passed is False

    async def test_handles_exception_during_test_execution(self, mock_client):
        tester = TesterAgent(client=mock_client)
        tester._run_tests = AsyncMock(side_effect=RuntimeError("boom"))
        result = await tester.run("goal", [])
        assert isinstance(result, TestResult)
        assert result.passed is False
        assert "boom" in result.summary


class TestLLMClient:
    async def test_read_and_write_file(self, tmp_workspace):
        client = LLMClient(settings=Settings(workspace=tmp_workspace))
        await client.write_file("sub/deep.txt", "contents")
        assert (tmp_workspace / "sub" / "deep.txt").read_text() == "contents"
        assert await client.read_file("sub/deep.txt") == "contents"

    def test_list_files_returns_relative_paths(self, tmp_workspace):
        (tmp_workspace / "a.txt").write_text("a")
        (tmp_workspace / "sub").mkdir()
        (tmp_workspace / "sub" / "b.txt").write_text("b")
        client = LLMClient(settings=Settings(workspace=tmp_workspace))
        files = sorted(client.list_files(tmp_workspace))
        assert files == ["a.txt", "sub/b.txt"]

    async def test_unsupported_provider_raises_value_error(self, tmp_workspace):
        client = LLMClient(settings=Settings(workspace=tmp_workspace))
        client.settings.provider = "unsupported"
        with pytest.raises(tenacity.RetryError) as exc_info:
            await client.complete("prompt")
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "Unsupported provider" in str(exc_info.value.__cause__)
