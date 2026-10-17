"""Tests for the Furrow orchestrator, agents, and LLM client.

These tests use a mock LLM client so no API key or network access is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class MockLLMClient(LLMClient):
    """LLMClient stand-in that returns canned responses and needs no API key."""

    def __init__(self, complete_response: str = "") -> None:
        # Avoid calling LLMClient.__init__ which touches settings/env.
        self.complete_response = complete_response
        self.calls: list[tuple[str, str, str | None]] = []
        self.settings = Settings()
        self.read_files: list[str] = []
        self.written_files: dict[str, str] = {}

    def validate(self) -> None:
        pass

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append((prompt, system, model))
        return self.complete_response

    async def read_file(self, path: str | Path) -> str:
        self.read_files.append(str(path))
        return ""

    async def write_file(self, path: str | Path, content: str) -> None:
        self.written_files[str(path)] = content

    def list_files(self, directory: str | Path) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def make_orchestrator(
    goal: str = "test goal", max_cycles: int = 0, state_file: Path | None = None
) -> Orchestrator:
    client = MockLLMClient()
    orch = Orchestrator(goal=goal, client=client, state_file=state_file)
    orch.settings.max_cycles = max_cycles
    return orch


def test_is_done_empty_tasks_is_done():
    orch = make_orchestrator()
    assert orch._is_done() is True


def test_is_done_all_completed_is_done():
    orch = make_orchestrator()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_any_failed_is_not_done():
    orch = make_orchestrator()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_pending_is_not_done():
    orch = make_orchestrator()
    orch.tasks = [TaskModel(id="1", description="a", status="pending")]
    assert orch._is_done() is False


def test_get_tasks_returns_persistent_list():
    orch = make_orchestrator()
    task = TaskModel(id="1", description="a")
    orch.tasks.append(task)
    assert orch._get_tasks() == [task]


def test_cycle_limit_halted(capsys):
    orch = make_orchestrator(max_cycles=1)
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    import asyncio

    asyncio.run(orch.run())
    assert orch.cycles == 1


def test_plan_merge_adds_new_tasks_by_id():
    orch = make_orchestrator()
    existing = TaskModel(id="1", description="existing", status="completed")
    orch.tasks.append(existing)

    new_plan = Plan(
        tasks=[
            TaskModel(id="1", description="existing updated"),
            TaskModel(id="2", description="new task"),
        ],
        rationale="x",
    )
    orch.planner.plan = AsyncMock(return_value=new_plan)

    import asyncio

    asyncio.run(orch._cycle())
    ids = [t.id for t in orch.tasks]
    assert ids == ["1", "2"]
    # Existing task kept its status (not clobbered by the new plan entry).
    assert orch.tasks[0].status == "completed"


def test_dependency_resolution_runs_ready_first():
    orch = make_orchestrator()
    orch.tasks = [
        TaskModel(id="a", description="first", status="pending", dependencies=[]),
        TaskModel(id="b", description="second", status="pending", dependencies=["a"]),
    ]
    # MockLLMClient.complete returns "" (not an exception), so the orchestrator
    # marks each executed task as completed. Task "b" must wait for "a".
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="x"))
    import asyncio

    asyncio.run(orch._cycle())
    assert orch.tasks[0].status == "completed"
    assert orch.tasks[1].status == "completed"


def test_state_save_and_load(tmp_path):
    state_file = tmp_path / "state.json"
    orch = make_orchestrator(goal="persisted goal", state_file=state_file)
    orch.cycles = 3
    orch.tasks = [TaskModel(id="1", description="a", status="completed", result="ok")]
    orch.save_state()

    loaded = Orchestrator(goal="fresh", client=MockLLMClient(), state_file=state_file)
    loaded.settings = orch.settings
    assert loaded.goal == "persisted goal"
    assert loaded.cycles == 3
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].status == "completed"


# ---------------------------------------------------------------------------
# Worker agent
# ---------------------------------------------------------------------------


def test_worker_writes_files_from_marker_response(tmp_path):
    client = MockLLMClient(
        complete_response="### FILE: hello.py\nprint('hello')\n### FILE: world.py\nprint('world')\n"
    )
    worker = WorkerAgent(
        task=TaskModel(id="1", description="write files", files=["hello.py"]),
        client=client,
    )
    # Point the workspace at tmp_path so files land there.
    import os

    orig_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        import asyncio

        asyncio.run(worker.run())
        assert (tmp_path / "hello.py").read_text() == "print('hello')"
        assert (tmp_path / "world.py").read_text() == "print('world')"
        assert "hello.py" in worker.written_files
    finally:
        os.chdir(orig_cwd)


def test_worker_reads_existing_files():
    client = MockLLMClient(complete_response="summary")
    worker = WorkerAgent(
        task=TaskModel(id="1", description="edit", files=["src/existing.py"]),
        client=client,
    )
    import asyncio

    asyncio.run(worker.run())
    assert "src/existing.py" in client.read_files


# ---------------------------------------------------------------------------
# Planner agent
# ---------------------------------------------------------------------------


def test_planner_parses_valid_plan():
    client = MockLLMClient(
        complete_response='{"tasks":[{"id":"1","description":"do x","files":["a.py"]}],"rationale":"ok"}'
    )
    planner = PlannerAgent(client=client)
    import asyncio

    plan = asyncio.run(planner.plan("do x"))
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "ok"


def test_planner_raises_on_invalid_json():
    client = MockLLMClient(complete_response="not json at all")
    planner = PlannerAgent(client=client)
    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(planner.plan("do x"))


# ---------------------------------------------------------------------------
# Tester agent
# ---------------------------------------------------------------------------


def test_tester_parses_passed_result():
    client = MockLLMClient(complete_response='{"passed":true,"summary":"all green","failures":[]}')
    tester = TesterAgent(client=client)
    import asyncio

    result = asyncio.run(tester.run("goal", []))
    assert result.passed is True
    assert result.summary == "all green"


def test_tester_falls_back_to_substring_match():
    client = MockLLMClient(complete_response="All tests passed successfully")
    tester = TesterAgent(client=client)
    import asyncio

    result = asyncio.run(tester.run("goal", []))
    assert result.passed is True


def test_tester_reports_failure_when_no_tests():
    client = MockLLMClient(complete_response="No test runner found.")
    tester = TesterAgent(client=client)
    import asyncio

    result = asyncio.run(tester.run("goal", []))
    assert result.passed is False
