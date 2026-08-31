from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult


def _make_client():
    client = AsyncMock()
    client.settings = Settings()
    return client


class TestPlannerAgent:
    async def test_plan_with_valid_json(self):
        client = _make_client()
        task_data = [
            {"id": "1", "description": "first task"},
            {"id": "2", "description": "second task"},
        ]
        response = json.dumps({"tasks": task_data, "rationale": "build it"})
        client.complete = AsyncMock(return_value=response)

        agent = PlannerAgent(client=client)
        plan = await agent.plan("build a web app")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.rationale == "build it"

    async def test_plan_uses_planner_model(self):
        client = _make_client()
        client.complete = AsyncMock(return_value=json.dumps({"tasks": [], "rationale": "ok"}))

        agent = PlannerAgent(client=client)
        await agent.plan("do stuff")

        client.complete.assert_called_once()
        call_kwargs = client.complete.call_args
        assert call_kwargs.kwargs["model"] == client.settings.planner_model

    async def test_plan_invalid_json_raises(self):
        client = _make_client()
        client.complete = AsyncMock(return_value="not json")

        agent = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("build something")

    async def test_plan_missing_fields_raises(self):
        client = _make_client()
        client.complete = AsyncMock(return_value=json.dumps({"bad": "data"}))

        agent = PlannerAgent(client=client)
        with pytest.raises(ValueError):
            await agent.plan("build something")


class TestWorkerAgent:
    async def test_run_with_valid_response(self, tmp_path):
        client = _make_client()
        client.settings.workspace = tmp_path
        client.complete = AsyncMock(
            return_value=json.dumps({
                "files_modified": [{"path": "main.py", "content": "print('hi')"}],
                "summary": "done",
                "success": True,
            })
        )

        task = TaskModel(id="1", description="write main.py", files=["main.py"])
        agent = WorkerAgent(task=task, client=client)
        result = await agent.run()

        assert result["success"] is True
        assert result["summary"] == "done"
        assert result["files_modified"] == ["main.py"]
        assert (tmp_path / "main.py").exists()

    async def test_run_writes_file_to_workspace(self, tmp_path):
        client = _make_client()
        client.settings.workspace = tmp_path
        client.complete = AsyncMock(
            return_value=json.dumps({
                "files_modified": [{"path": "src/app.py", "content": "code"}],
                "summary": "ok",
                "success": True,
            })
        )

        task = TaskModel(id="1", description="write code")
        agent = WorkerAgent(task=task, client=client)
        await agent.run()

        assert (tmp_path / "src" / "app.py").exists()
        assert (tmp_path / "src" / "app.py").read_text() == "code"

    async def test_run_with_invalid_json_returns_failure(self):
        client = _make_client()
        client.complete = AsyncMock(return_value="not json")

        task = TaskModel(id="1", description="do work")
        agent = WorkerAgent(task=task, client=client)
        result = await agent.run()

        assert result["success"] is False
        assert result["files_modified"] == []
        assert "Failed to parse" in result["summary"]

    async def test_run_skips_missing_files(self):
        client = _make_client()
        client.complete = AsyncMock(
            return_value=json.dumps({
                "files_modified": [],
                "summary": "no files",
                "success": True,
            })
        )

        task = TaskModel(id="1", description="do work", files=["missing.py"])
        agent = WorkerAgent(task=task, client=client)
        result = await agent.run()

        assert result["success"] is True
        assert result["files_modified"] == []


class TestTesterAgent:
    async def test_run_with_valid_result(self):
        client = _make_client()
        client.complete = AsyncMock(
            return_value=json.dumps({
                "passed": True,
                "summary": "all tests pass",
                "failures": [],
            })
        )

        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="task")]
        result = await agent.run("build app", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "all tests pass"

    async def test_run_with_invalid_json_fallback(self):
        client = _make_client()
        client.complete = AsyncMock(return_value="something about passed")

        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="task")]
        result = await agent.run("build app", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert "something about passed" in result.summary

    async def test_run_invalid_json_no_passed_keyword(self):
        client = _make_client()
        client.complete = AsyncMock(return_value="completely invalid response")

        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="task")]
        result = await agent.run("build app", tasks)

        assert result.passed is False
        assert result.failures == []

    async def test_run_exception_in_test_execution(self):
        client = _make_client()
        client.complete = AsyncMock()

        agent = TesterAgent(client=client)
        agent._run_tests = AsyncMock(side_effect=RuntimeError("test runner broke"))

        tasks = [TaskModel(id="1", description="task")]
        result = await agent.run("build app", tasks)

        assert result.passed is False
        assert "test runner broke" in result.summary
