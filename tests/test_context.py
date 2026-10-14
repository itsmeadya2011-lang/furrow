import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.context import ContextManager, FileSummary


class FakeSettings:
    worker_model = "test-worker-model"
    planner_model = "test-planner-model"
    tester_model = "test-tester-model"


class FakeLLMClient:
    """Minimal stand-in for LLMClient that returns canned responses."""

    def __init__(self, response: str = '{"changes": [], "summary": "done"}') -> None:
        self.settings = FakeSettings()
        self.response = response
        self.calls: list[tuple[str, str]] = []
        self._read_store: dict[str, str] = {}
        self._write_store: dict[str, str] = {}

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append((prompt, model or self.settings.worker_model))
        return self.response

    async def complete_with_usage(self, prompt: str, system: str = "", model: str | None = None):
        content = await self.complete(prompt, system, model)
        from furrow.llm import LLMResponse, TokenUsage

        return LLMResponse(content=content, usage=TokenUsage(), model=model or "")

    async def read_file(self, path: str | Path) -> str:
        key = str(path)
        if key not in self._read_store:
            raise FileNotFoundError(key)
        return self._read_store[key]

    async def write_file(self, path: str | Path, content: str) -> None:
        self._write_store[str(path)] = content


def make_context(tmp_path: Path) -> tuple[ContextManager, FakeLLMClient]:
    client = FakeLLMClient()
    return ContextManager(workspace=tmp_path, client=client), client


# ---------------------------------------------------------------- #
# ContextManager unit tests
# ---------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_update_from_changes_stores_file_summary(tmp_path):
    ctx, _client = make_context(tmp_path)
    changes = [{"file": "src/main.py", "content": "def main(): pass"}]
    await ctx.update_from_changes(changes, cycle=1)

    assert "src/main.py" in ctx.file_summaries
    summary = ctx.file_summaries["src/main.py"]
    assert summary.path == "src/main.py"
    assert summary.last_modified_cycle == 1
    assert summary.lines_changed == 1
    assert len(summary.summary) > 0


@pytest.mark.asyncio
async def test_update_from_changes_stores_recent_changes(tmp_path):
    ctx, _ = make_context(tmp_path)
    changes = [{"file": "src/main.py", "content": "old\nnew\n"}]
    await ctx.update_from_changes(changes, cycle=2)

    assert len(ctx.recent_changes) == 1
    entry = ctx.recent_changes[0]
    assert entry["cycle"] == 2
    assert entry["file"] == "src/main.py"
    assert "timestamp" in entry


@pytest.mark.asyncio
async def test_update_from_changes_skips_missing_file_field(tmp_path):
    ctx, _ = make_context(tmp_path)
    changes = [{"content": "data"}]
    await ctx.update_from_changes(changes, cycle=1)

    assert ctx.file_summaries == {}
    assert ctx.recent_changes == []


@pytest.mark.asyncio
async def test_summarize_file_fallback_on_exception(tmp_path):
    client = FakeLLMClient()
    client.complete = AsyncMock(side_effect=RuntimeError("API down"))
    ctx = ContextManager(workspace=tmp_path, client=client)

    content = "line1\nline2\nline3"
    summary = await ctx._summarize_file("test.py", content)

    assert "3 lines" in summary
    assert f"{len(content)} chars" in summary


@pytest.mark.asyncio
async def test_summarize_file_empty_content(tmp_path):
    ctx, _ = make_context(tmp_path)
    summary = await ctx._summarize_file("empty.py", "")
    assert summary == "Empty file"


@pytest.mark.asyncio
async def test_recent_changes_trimmed(tmp_path):
    ctx, _ = make_context(tmp_path)
    ctx.max_recent_changes = 3

    for i in range(5):
        await ctx.update_from_changes([{"file": f"f{i}.py", "content": "x"}], cycle=i)

    assert len(ctx.recent_changes) == 3
    # Most recent entries should be the last three cycles
    assert ctx.recent_changes[-1]["cycle"] == 4
    assert ctx.recent_changes[0]["cycle"] == 2


@pytest.mark.asyncio
async def test_get_context_for_planner(tmp_path):
    ctx, _ = make_context(tmp_path)
    await ctx.update_from_changes([{"file": "a.py", "content": "a"}], cycle=1)
    ctx.add_cycle_summary("Built feature X")

    result = ctx.get_context_for_planner()

    assert "## Project Files" in result
    assert "a.py" in result
    assert "## Recent Cycles" in result
    assert "Built feature X" in result
    assert "## Recent Changes" in result


def test_get_context_for_planner_empty(tmp_path):
    ctx = ContextManager(workspace=tmp_path, client=FakeLLMClient())
    assert ctx.get_context_for_planner() == "No context available."


def test_add_cycle_summary_trims(tmp_path):
    ctx = ContextManager(workspace=tmp_path, client=FakeLLMClient())
    ctx.max_cycle_summaries = 3
    for i in range(5):
        ctx.add_cycle_summary(f"summary {i}")
    assert len(ctx.cycle_summaries) == 3
    assert ctx.cycle_summaries[-1] == "summary 4"


def test_get_context_for_worker_relevant_files(tmp_path):
    ctx = ContextManager(workspace=tmp_path, client=FakeLLMClient())
    ctx.file_summaries["a.py"] = FileSummary(
        path="a.py", summary="summary-a", last_modified_cycle=0, lines_changed=1
    )
    ctx.file_summaries["b.py"] = FileSummary(
        path="b.py", summary="summary-b", last_modified_cycle=0, lines_changed=2
    )

    result = ctx.get_context_for_worker(["a.py"])

    assert "## Relevant File Summaries" in result
    assert "a.py" in result
    assert "summary-a" in result


def test_get_context_for_worker_no_relevant(tmp_path):
    ctx = ContextManager(workspace=tmp_path, client=FakeLLMClient())
    assert ctx.get_context_for_worker(["missing.py"]) == "No specific context."


def test_context_manager_initial_state(tmp_path):
    ctx, client = make_context(tmp_path)
    assert ctx.workspace == tmp_path
    assert ctx.client is client
    assert ctx.file_summaries == {}
    assert ctx.recent_changes == []
    assert ctx.cycle_summaries == []
    assert ctx.max_recent_changes == 20
    assert ctx.max_cycle_summaries == 10


# ---------------------------------------------------------------- #
# WorkerAgent integration with ContextManager
# ---------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_worker_agent_uses_context(tmp_path):
    from furrow.agents.worker import WorkerAgent

    client = FakeLLMClient(
        response='{"changes": [{"file": "out.py", "content": "print(1)"}], "summary": "wrote out"}'
    )
    ctx = ContextManager(workspace=tmp_path, client=client)
    # Seed a file summary so worker context includes it
    await ctx.update_from_changes([{"file": "out.py", "content": "old"}], cycle=1)

    task = TaskModel(id="t1", description="create out.py", files=["out.py"])
    worker = WorkerAgent(task=task, client=client, context=ctx)
    result = await worker.run()

    data = json.loads(result)
    assert data["summary"] == "wrote out"
    # The file was written via the LLM client
    assert client._write_store["out.py"] == "print(1)"
    # The context was consulted for the worker (last call is the worker prompt)
    assert "## Relevant File Summaries" in client.calls[-1][0]


@pytest.mark.asyncio
async def test_worker_agent_without_context(tmp_path):
    from furrow.agents.worker import WorkerAgent

    client = FakeLLMClient(response='{"changes": [], "summary": "noop"}')
    task = TaskModel(id="t2", description="noop", files=[])
    worker = WorkerAgent(task=task, client=client, context=None)
    result = await worker.run()

    data = json.loads(result)
    assert data["summary"] == "noop"


# ---------------------------------------------------------------- #
# Orchestrator integration with ContextManager
# ---------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_orchestrator_uses_context_manager(tmp_path, monkeypatch):
    """The orchestrator should instantiate and drive the ContextManager during a cycle."""
    from furrow.core.orchestrator import Orchestrator

    # Fake planner that returns a simple plan with one task
    plan = Plan(
        tasks=[TaskModel(id="task-1", description="do work", files=["x.py"])], rationale="ok"
    )
    planner_mock = MagicMock()
    planner_mock.plan = AsyncMock(return_value=plan)

    # Fake tester that passes
    tester_mock = MagicMock()
    tester_instance = tester_mock.return_value
    tester_instance.run = AsyncMock(
        return_value=TestResult(passed=True, summary="all good", failures=[])
    )

    # Fake worker result that yields changes
    worker_result = '{"changes": [{"file": "x.py", "content": "content"}], "summary": "built x"}'

    # Fake git manager
    git_mock = MagicMock()
    git_instance = git_mock.return_value
    git_instance.is_repo = AsyncMock(return_value=True)
    git_instance.get_initial_commit = AsyncMock(return_value="abc123")
    git_instance.save_checkpoint = AsyncMock(return_value="deadbeef")

    # Build orchestrator with fakes
    orchestrator = Orchestrator(goal="build something", client=FakeLLMClient(worker_result))
    orchestrator.planner = planner_mock
    orchestrator.git = git_instance
    orchestrator._git_initialized = True

    # Patch the WorkerAgent so we can spy on context being passed and control its run
    created_workers: list = []

    import furrow.core.orchestrator as orch_mod
    from furrow.agents.worker import WorkerAgent as RealWorkerAgent

    def worker_factory(task, client, context=None):
        w = RealWorkerAgent(task=task, client=client, context=context)
        w.run = AsyncMock(return_value=worker_result)
        created_workers.append(w)
        return w

    monkeypatch.setattr(orch_mod, "WorkerAgent", worker_factory)
    monkeypatch.setattr(orch_mod, "TesterAgent", tester_mock)

    ctx = orchestrator.context
    assert isinstance(ctx, ContextManager)
    assert ctx.client is orchestrator.client

    await orchestrator._cycle()

    # Context was updated with changes from the executed task
    assert "x.py" in ctx.file_summaries
    assert ctx.file_summaries["x.py"].last_modified_cycle == orchestrator.cycles

    # A cycle summary was added
    assert len(ctx.cycle_summaries) == 1
    assert "task-1" in created_workers[0].task.id
    # Worker received the context manager
    assert created_workers[0].context is ctx

    # Planner was called with planner context appended
    planner_mock.plan.assert_awaited_once()
    planner_prompt = planner_mock.plan.await_args[0][0]
    assert "Project Files" in planner_prompt or "No context available" in planner_prompt

    # Checkpoint saved via git manager
    git_instance.save_checkpoint.assert_awaited_once()
