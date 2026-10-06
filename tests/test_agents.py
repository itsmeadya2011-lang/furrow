from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult


# --------------------------------------------------------------------------- #
# PlannerAgent
# --------------------------------------------------------------------------- #


async def test_planner_json_parse_success() -> None:
    client = MagicMock()
    client.complete = AsyncMock(
        return_value='{"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}'
    )
    planner = PlannerAgent(client=client)

    plan = await planner.plan("make something")

    assert isinstance(plan, Plan)
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "ok"
    client.complete.assert_awaited_once()


async def test_planner_json_parse_failure() -> None:
    client = MagicMock()
    client.complete = AsyncMock(return_value="not json at all")
    planner = PlannerAgent(client=client)

    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("make something")


# --------------------------------------------------------------------------- #
# WorkerAgent
# --------------------------------------------------------------------------- #


async def test_worker_structured_output() -> None:
    task = TaskModel(id="1", description="implement feature", files=["src/app.py"])
    client = MagicMock()
    client.complete = AsyncMock(
        return_value=(
            '{"changes": [{"file": "src/app.py", "content": "print(1)"}], "summary": "done"}'
        )
    )
    client.write_file = AsyncMock()

    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert result == "done"
    client.write_file.assert_awaited_once()
    args, _ = client.write_file.call_args
    assert args[0] == "src/app.py"
    assert args[1] == "print(1)"


async def test_worker_fallback_on_bad_json() -> None:
    task = TaskModel(id="1", description="implement feature", files=["src/app.py"])
    client = MagicMock()
    client.complete = AsyncMock(return_value="plain text response, no json")
    client.write_file = AsyncMock()

    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert result == "plain text response, no json"
    client.write_file.assert_not_awaited()


# --------------------------------------------------------------------------- #
# TesterAgent
# --------------------------------------------------------------------------- #


async def test_tester_success() -> None:
    client = MagicMock()
    client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "All tests passed", "failures": []}'
    )
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="1 passed")

    result = await tester.run("do thing", [])

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


async def test_tester_json_parse() -> None:
    client = MagicMock()
    client.complete = AsyncMock(
        return_value='{"passed": false, "summary": "2 failed", "failures": ["a", "b"]}'
    )
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="failures")

    result = await tester.run("do thing", [])

    assert result.passed is False
    assert result.summary == "2 failed"
    assert result.failures == ["a", "b"]


async def test_tester_fallback() -> None:
    client = MagicMock()
    client.complete = AsyncMock(return_value="The tests passed successfully.")
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="ok")

    result = await tester.run("do thing", [])

    assert result.passed is True
    assert "passed" in result.summary.lower()


async def test_tester_narrow_exceptions() -> None:
    client = MagicMock()
    client.complete = AsyncMock(return_value="ignored")
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(side_effect=RuntimeError("boom"))

    result = await tester.run("do thing", [])

    assert result.passed is False
    assert "boom" in result.summary
    assert result.failures == ["boom"]