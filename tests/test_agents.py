"""Tests for furrow.agents prompts — PlannerAgent, WorkerAgent, TesterAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.prompts import (
    PLANNER_PROMPT,
    TESTER_PROMPT,
    WORKER_PROMPT,
)
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


def _settings() -> Settings:
    return Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")


def _client() -> LLMClient:
    return LLMClient(settings=_settings())


class TestPlannerPrompt:
    def test_planner_prompt_format(self) -> None:
        """Verify planner prompt contains expected text."""
        assert "planning agent" in PLANNER_PROMPT
        assert "Furrow" in PLANNER_PROMPT
        assert "parallelizable" in PLANNER_PROMPT
        assert "tasks" in PLANNER_PROMPT
        assert "rationale" in PLANNER_PROMPT
        assert "JSON" in PLANNER_PROMPT
        # Should mention both tasks and rationale keys
        assert '"tasks"' in PLANNER_PROMPT
        assert '"rationale"' in PLANNER_PROMPT

    @pytest.mark.asyncio
    async def test_planner_parses_json_response(self) -> None:
        """PlannerAgent.plan parses JSON and returns a Plan."""
        client = _client()
        payload = {
            "tasks": [
                {"id": "1", "description": "do x", "files": [], "dependencies": []}
            ],
            "rationale": "because",
        }
        client.complete = AsyncMock(return_value=json.dumps(payload))  # type: ignore[method-assign]

        agent = PlannerAgent(client=client)
        plan = await agent.plan("the goal")
        assert isinstance(plan, Plan)
        assert plan.rationale == "because"
        assert plan.tasks[0].id == "1"
        # And complete was called with the planner prompt + goal
        client.complete.assert_awaited_once()
        args, _kwargs = client.complete.call_args
        assert "Goal: the goal" in args[0]
        assert PLANNER_PROMPT in args[0]

    @pytest.mark.asyncio
    async def test_planner_raises_on_invalid_json(self) -> None:
        client = _client()
        client.complete = AsyncMock(return_value="not json")  # type: ignore[method-assign]
        agent = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("g")


class TestWorkerPrompt:
    def test_worker_prompt_format(self) -> None:
        """Verify worker prompt contains expected text."""
        assert "worker agent" in WORKER_PROMPT
        assert "Furrow" in WORKER_PROMPT
        assert "minimal" in WORKER_PROMPT.lower() or "targeted" in WORKER_PROMPT.lower()
        assert "subagents" in WORKER_PROMPT or "spawn" in WORKER_PROMPT

    @pytest.mark.asyncio
    async def test_worker_run_includes_task_description(self) -> None:
        client = _client()
        client.complete = AsyncMock(return_value="done")  # type: ignore[method-assign]

        task = TaskModel(
            id="1",
            description="implement foo",
            files=["src/foo.py"],
        )
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()
        assert result == "done"
        client.complete.assert_awaited_once()
        args, _kwargs = client.complete.call_args
        assert "Task: implement foo" in args[0]
        assert "src/foo.py" in args[0]
        assert WORKER_PROMPT in args[0]

    @pytest.mark.asyncio
    async def test_worker_run_with_no_files(self) -> None:
        client = _client()
        client.complete = AsyncMock(return_value="ok")  # type: ignore[method-assign]
        task = TaskModel(id="1", description="x")
        worker = WorkerAgent(task=task, client=client)
        await worker.run()
        args, _ = client.complete.call_args
        assert "any" in args[0]


class TestTesterPrompt:
    def test_tester_prompt_format(self) -> None:
        """Verify tester prompt contains expected text."""
        assert "tester agent" in TESTER_PROMPT
        assert "Furrow" in TESTER_PROMPT
        assert "JSON" in TESTER_PROMPT
        assert "passed" in TESTER_PROMPT
        assert "failures" in TESTER_PROMPT

    @pytest.mark.asyncio
    async def test_tester_parses_json_response(self) -> None:
        client = _client()
        client.complete = AsyncMock(  # type: ignore[method-assign]
            return_value=json.dumps(
                {"passed": True, "summary": "all good", "failures": []}
            )
        )
        agent = TesterAgent(client=client)
        # Patch out the test runner to avoid touching the host filesystem
        agent._run_tests = AsyncMock(return_value="1 passed")  # type: ignore[method-assign]
        result = await agent.run("the goal", [])
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "all good"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_tester_handles_non_json_response(self) -> None:
        client = _client()
        client.complete = AsyncMock(return_value="all passed, no issues")  # type: ignore[method-assign]
        agent = TesterAgent(client=client)
        agent._run_tests = AsyncMock(return_value="ok")  # type: ignore[method-assign]
        result = await agent.run("g", [])
        assert result.passed is True
        assert "passed" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_tester_handles_test_runner_error(self) -> None:
        client = _client()
        agent = TesterAgent(client=client)
        agent._run_tests = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        result = await agent.run("g", [])
        assert result.passed is False
        assert "boom" in result.summary