"""Tests for the Orchestrator in furrow.core.orchestrator.

A FakeLLMClient returns canned JSON so no real API keys are required.

The real TesterAgent spawns subprocesses (pytest, npm, ...) which would
recurse into the test suite and hang, so the orchestrator tests monkeypatch
``TesterAgent.run`` to return a controlled TestResult instead.

To exercise the ``max_cycles`` cap the worker must keep failing (otherwise
``_is_done()`` returns True after cycle 1 and the loop halts early). The
FakeLLMClient therefore returns a valid plan for planner prompts and raises
for worker prompts (distinguished by the presence of "Task:" in the prompt),
which the orchestrator records as a failed task.
"""

import asyncio
import json
from pathlib import Path

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakeLLMClient:
    """Returns a valid plan for planner prompts, raises for worker prompts."""

    def __init__(self, plan_json: str) -> None:
        self.plan_json = plan_json
        self.complete_calls = 0
        # Provide a settings attribute so agents constructed by the
        # orchestrator (WorkerAgent) can read model names.
        self.settings = Settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls += 1
        # The planner prompt contains "Goal:"; the worker prompt contains
        # "Task:". Distinguish by content so repeated cycles work correctly.
        if "Task:" in prompt:
            raise RuntimeError("worker boom")
        return self.plan_json


def _plan_json_with_failing_task() -> str:
    return json.dumps(
        {
            "tasks": [
                {
                    "id": "1",
                    "description": "write a broken script",
                    "files": ["broken.py"],
                    "dependencies": [],
                }
            ],
            "rationale": "one failing task so the loop keeps going",
        }
    )


def _failing_test_result() -> TestResult:
    return TestResult(passed=False, summary="tests broke", failures=["broken.py is wrong"])


def test_max_cycles_enforced() -> None:
    """The orchestrator halts after settings.max_cycles cycles."""
    from furrow.core import orchestrator as orchestrator_module
    from furrow.agents import tester as tester_module

    client = FakeLLMClient(_plan_json_with_failing_task())
    orchestrator = Orchestrator(goal="fix it", client=client)

    async def fake_run(self, goal, tasks):
        return _failing_test_result()

    original_run = tester_module.TesterAgent.run
    tester_module.TesterAgent.run = fake_run
    original_max = orchestrator_module.settings.max_cycles
    orchestrator_module.settings.max_cycles = 2
    try:
        asyncio.run(orchestrator.run())
    finally:
        tester_module.TesterAgent.run = original_run
        orchestrator_module.settings.max_cycles = original_max

    assert orchestrator.cycles == 2
    # The mutable goal should have been rewritten to reflect the failures.
    assert "Fix failing tests" in orchestrator.goal
    # The original goal is untouched.
    assert orchestrator.original_goal == "fix it"


def test_original_goal_preserved_separately_from_mutable_goal() -> None:
    """The original goal is stored on the instance and never mutated."""
    client = FakeLLMClient(_plan_json_with_failing_task())
    orchestrator = Orchestrator(goal="build a widget", client=client)

    assert orchestrator.original_goal == "build a widget"
    assert orchestrator.goal == "build a widget"

    # Simulate the mutation the orchestrator performs on test failure.
    orchestrator.goal = "Fix failing tests:\nbroken thing"

    assert orchestrator.original_goal == "build a widget"
    assert orchestrator.goal != orchestrator.original_goal


def test_is_done_true_when_all_completed() -> None:
    client = FakeLLMClient(_plan_json_with_failing_task())
    orchestrator = Orchestrator(goal="do thing", client=client)

    task = TaskModel(id="1", description="task", status="completed")
    orchestrator.plan = Plan(tasks=[task], rationale="r")

    assert orchestrator._is_done() is True


def test_is_done_false_when_any_failed() -> None:
    client = FakeLLMClient(_plan_json_with_failing_task())
    orchestrator = Orchestrator(goal="do thing", client=client)

    failed = TaskModel(id="1", description="task", status="failed")
    orchestrator.plan = Plan(tasks=[failed], rationale="r")

    assert orchestrator._is_done() is False


def test_is_done_false_when_pending() -> None:
    client = FakeLLMClient(_plan_json_with_failing_task())
    orchestrator = Orchestrator(goal="do thing", client=client)

    pending = TaskModel(id="1", description="task", status="pending")
    orchestrator.plan = Plan(tasks=[pending], rationale="r")

    assert orchestrator._is_done() is False


def test_is_done_false_when_no_plan() -> None:
    client = FakeLLMClient(_plan_json_with_failing_task())
    orchestrator = Orchestrator(goal="do thing", client=client)

    assert orchestrator.plan is None
    assert orchestrator._is_done() is False


def test_is_done_false_when_no_tasks() -> None:
    client = FakeLLMClient(_plan_json_with_failing_task())
    orchestrator = Orchestrator(goal="do thing", client=client)

    orchestrator.plan = Plan(tasks=[], rationale="empty")

    assert orchestrator._is_done() is False


def test_run_stops_when_all_tasks_completed(tmp_path: Path) -> None:
    """If the worker succeeds and tests pass, the orchestrator halts after one cycle."""
    from furrow.core import orchestrator as orchestrator_module
    from furrow.agents import tester as tester_module

    plan_json = json.dumps(
        {
            "tasks": [
                {
                    "id": "1",
                    "description": "write a good script",
                    "files": ["good.py"],
                    "dependencies": [],
                }
            ],
            "rationale": "one good task",
        }
    )

    class GoodClient(FakeLLMClient):
        """Returns the plan then a valid code block so the worker succeeds."""

        async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
            self.complete_calls += 1
            if "Task:" in prompt:
                return "```python\ngood.py\nprint('ok')\n```"
            return self.plan_json

    client = GoodClient(plan_json)
    client.settings.workspace = tmp_path
    orchestrator = Orchestrator(goal="be good", client=client)

    async def fake_run(self, goal, tasks):
        return TestResult(passed=True, summary="all good", failures=[])

    original_run = tester_module.TesterAgent.run
    tester_module.TesterAgent.run = fake_run
    original_max = orchestrator_module.settings.max_cycles
    orchestrator_module.settings.max_cycles = 0
    try:
        asyncio.run(orchestrator.run())
    finally:
        tester_module.TesterAgent.run = original_run
        orchestrator_module.settings.max_cycles = original_max

    assert orchestrator.cycles == 1
    assert orchestrator.plan is not None
    assert orchestrator.plan.tasks[0].status == "completed"