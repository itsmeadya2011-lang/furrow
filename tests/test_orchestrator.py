"""Tests for the Orchestrator class.

These tests use an in-memory ``FakeLLMClient`` (no API keys required) and
``unittest.mock`` to patch the agent classes so the orchestrator can be driven
without making real LLM calls.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Provider, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakeSettings:
    """Deterministic stand-in for ``furrow.config.Settings``."""

    def __init__(self, max_cycles: int = 0) -> None:
        self.provider = Provider.ANTHROPIC
        self.model = "claude-sonnet-4-20250514"
        self.planner_model = "claude-3-5-haiku-20241022"
        self.worker_model = "claude-3-5-sonnet-20241022"
        self.tester_model = "claude-3-5-sonnet-20241022"
        self.anthropic_api_key = None
        self.openai_api_key = None
        self.ollama_base_url = "http://localhost:11434"
        self.max_parallel_tasks = 5
        self.max_cycles = max_cycles
        self.workspace = Path.cwd()
        self.log_level = "INFO"


class FakeLLMClient:
    """In-memory substitute for ``furrow.llm.LLMClient``.

    ``complete`` returns pre-set responses one per call (configurable per call
    by passing a list of return values or exceptions) and works with an
    in-memory dict for file operations.
    """

    def __init__(self, settings=None, responses=None) -> None:
        self.settings = settings if settings is not None else FakeSettings()
        self._responses = list(responses) if responses else []
        self._call_index = 0
        self.files: dict[str, str] = {}

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        timeout: float = 60.0,
    ) -> str:
        if not self._responses:
            return ""
        if self._call_index < len(self._responses):
            response = self._responses[self._call_index]
        else:
            # Replay the last preset response when exhausted.
            response = self._responses[-1]
        self._call_index += 1
        if isinstance(response, BaseException):
            raise response
        return response

    async def read_file(self, path) -> str:
        key = str(path)
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key]

    async def write_file(self, path, content: str) -> None:
        self.files[str(path)] = content

    async def edit_file(self, path, old_string: str, new_string: str) -> bool:
        key = str(path)
        content = self.files.get(key, "")
        if old_string not in content:
            return False
        self.files[key] = content.replace(old_string, new_string, 1)
        return True

    async def file_exists(self, path) -> bool:
        return str(path) in self.files

    def list_files(self, directory) -> list[str]:
        prefix = str(directory)
        return sorted(name for name in self.files if name.startswith(prefix))


@pytest.fixture
def fake_plan() -> Plan:
    return Plan(
        tasks=[TaskModel(id="t1", description="Implement feature X", status="pending")],
        rationale="Need feature X",
    )


@pytest.fixture
def agents():
    """Patch the agent classes the Orchestrator depends on."""
    with (
        patch("furrow.core.orchestrator.PlannerAgent") as PlannerMock,
        patch("furrow.core.orchestrator.TesterAgent") as TesterMock,
        patch("furrow.core.orchestrator.WorkerAgent") as WorkerMock,
    ):
        planner_instance = PlannerMock.return_value
        tester_instance = TesterMock.return_value
        worker_instance = WorkerMock.return_value
        planner_instance.plan = AsyncMock()
        tester_instance.run = AsyncMock()
        worker_instance.run = AsyncMock()
        yield {
            "PlannerAgent": PlannerMock,
            "TesterAgent": TesterMock,
            "WorkerAgent": WorkerMock,
            "planner": planner_instance,
            "tester": tester_instance,
            "worker": worker_instance,
        }


# ---------------------------------------------------------------------------
# _is_done() logic
# ---------------------------------------------------------------------------


def test_is_done_all_completed():
    orch = Orchestrator(goal="g", client=FakeLLMClient())
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_some_pending():
    orch = Orchestrator(goal="g", client=FakeLLMClient())
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


def test_is_done_any_failed():
    orch = Orchestrator(goal="g", client=FakeLLMClient())
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_empty_tasks():
    orch = Orchestrator(goal="g", client=FakeLLMClient())
    orch.tasks = []
    assert orch._is_done() is True


# ---------------------------------------------------------------------------
# Task tracking
# ---------------------------------------------------------------------------


def test_get_tasks_returns_self_tasks():
    orch = Orchestrator(goal="g", client=FakeLLMClient())
    tasks = [TaskModel(id="1", description="a")]
    orch.tasks = tasks
    assert orch._get_tasks() is tasks


async def test_cycle_populates_tasks_from_plan(agents, fake_plan):
    orch = Orchestrator(goal="build a thing", client=FakeLLMClient())
    agents["planner"].plan = AsyncMock(return_value=fake_plan)
    agents["tester"].run = AsyncMock(return_value=TestResult(passed=True, summary="all good"))

    await orch._cycle()

    assert orch.tasks == fake_plan.tasks
    assert orch.tasks[0] is fake_plan.tasks[0]


# ---------------------------------------------------------------------------
# max_cycles enforcement
# ---------------------------------------------------------------------------


async def test_max_cycles_stops_after_two_cycles(agents, fake_plan):
    settings = FakeSettings(max_cycles=2)
    orch = Orchestrator(goal="build a thing", client=FakeLLMClient(settings=settings))
    assert orch.max_cycles == 2

    agents["planner"].plan = AsyncMock(return_value=fake_plan)
    # Worker always fails so tasks stay "failed" -> _is_done() stays False,
    # forcing the loop to rely on the max_cycles cap instead of early exit.
    agents["worker"].run = AsyncMock(side_effect=RuntimeError("boom"))
    agents["tester"].run = AsyncMock(return_value=TestResult(passed=True, summary="ok"))

    await orch.run()

    assert orch.cycles == 2
    assert agents["planner"].plan.call_count == 2


def test_max_cycles_zero_means_infinite():
    orch = Orchestrator(goal="g", client=FakeLLMClient(FakeSettings(max_cycles=0)))
    assert orch.max_cycles == 0


# ---------------------------------------------------------------------------
# Progress callback (_emit)
# ---------------------------------------------------------------------------


def test_emit_calls_progress_callback():
    calls: list[str] = []
    orch = Orchestrator(goal="g", client=FakeLLMClient(), progress_callback=calls.append)
    orch._emit("cycle starting")
    orch._emit("task done")
    assert calls == ["cycle starting", "task done"]


def test_emit_swallows_callback_exception():
    def bad_callback(status: str) -> None:
        raise ValueError("callback exploded")

    orch = Orchestrator(goal="g", client=FakeLLMClient(), progress_callback=bad_callback)
    # Must not raise even though the callback explodes.
    orch._emit("something happened")


# ---------------------------------------------------------------------------
# _execute_task() retry logic
# ---------------------------------------------------------------------------


async def test_execute_task_succeeds_first_attempt(agents):
    orch = Orchestrator(goal="g", client=FakeLLMClient())
    agents["worker"].run = AsyncMock(return_value="done")
    task = TaskModel(id="t1", description="do thing", status="pending")

    await orch._execute_task(task)

    assert agents["worker"].run.call_count == 1
    assert task.status == "completed"
    assert task.result == "done"


async def test_execute_task_retries_on_failure_then_succeeds(agents):
    orch = Orchestrator(goal="g", client=FakeLLMClient())
    agents["worker"].run = AsyncMock(
        side_effect=[RuntimeError("fail 1"), RuntimeError("fail 2"), "recovered"]
    )
    task = TaskModel(id="t1", description="do thing", status="pending")

    await orch._execute_task(task)

    assert agents["worker"].run.call_count == 3
    assert task.status == "completed"
    assert task.result == "recovered"
    # Description should have been augmented with retry context.
    assert "Retry attempt 3/3" in task.description


async def test_execute_task_marks_failed_after_max_retries(agents):
    orch = Orchestrator(goal="g", client=FakeLLMClient())
    agents["worker"].run = AsyncMock(side_effect=RuntimeError("always fails"))
    task = TaskModel(id="t1", description="do thing", status="pending")

    await orch._execute_task(task)

    assert agents["worker"].run.call_count == 3
    assert task.status == "failed"


# ---------------------------------------------------------------------------
# FakeLLMClient sanity (validates the test fixture itself)
# ---------------------------------------------------------------------------


async def test_fake_llm_client_complete_ordering_and_replay():
    client = FakeLLMClient(responses=["first", "second", "third"])
    assert await client.complete("p1") == "first"
    assert await client.complete("p2") == "second"
    # Replays the last preset response when exhausted.
    assert await client.complete("p3") == "third"
    assert await client.complete("p4") == "third"


async def test_fake_llm_client_complete_raises():
    client = FakeLLMClient(responses=[RuntimeError("transient")])
    with pytest.raises(RuntimeError, match="transient"):
        await client.complete("p")


async def test_fake_llm_client_file_operations():
    client = FakeLLMClient()
    await client.write_file("foo/bar.py", "content")
    assert await client.file_exists("foo/bar.py")
    assert await client.read_file("foo/bar.py") == "content"
    assert await client.edit_file("foo/bar.py", "content", "replaced")
    assert await client.read_file("foo/bar.py") == "replaced"
    assert client.list_files("foo") == ["foo/bar.py"]
    assert await client.file_exists("missing.py") is False
    with pytest.raises(FileNotFoundError):
        await client.read_file("missing.py")
