from __future__ import annotations

import sys
from pathlib import Path

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


class FakeClient:
    def __init__(self, settings, responses):
        self.settings = settings
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        return self.responses.pop(0)

    async def read_file(self, path):
        return Path(path).read_text()


# ---------------------------------------------------------------- planner


async def test_planner_parses_fenced_json(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = FakeClient(
        settings,
        ['```json\n{"tasks": [{"id": "1", "description": "x"}], "rationale": "ok"}\n```'],
    )
    planner = PlannerAgent(client=client, settings=settings)
    plan = await planner.plan("goal")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "ok"


async def test_planner_retries_once(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = FakeClient(settings, ["garbage", '{"tasks": [], "rationale": "done"}'])
    planner = PlannerAgent(client=client, settings=settings)
    plan = await planner.plan("goal")
    assert len(plan.tasks) == 0
    assert len(client.prompts) == 2


async def test_planner_raises_after_two_failures(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = FakeClient(settings, ["bad", "worse"])
    planner = PlannerAgent(client=client, settings=settings)
    with pytest.raises(ValueError):
        await planner.plan("goal")


async def test_planner_includes_context_in_prompt(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = FakeClient(settings, ['{"tasks": [], "rationale": "done"}'])
    planner = PlannerAgent(client=client, settings=settings)
    await planner.plan("goal", context="did stuff")
    assert "did stuff" in client.prompts[0]
    assert "Work already done" in client.prompts[0]


# ----------------------------------------------------------------- worker


async def test_worker_parses_edits(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = FakeClient(
        settings,
        ['```json\n{"summary": "did it", "edits": [{"path": "a.py", "content": "x"}]}\n```'],
    )
    task = TaskModel(id="1", description="do thing", files=["a.py"])
    worker = WorkerAgent(task=task, client=client, settings=settings)
    result = await worker.run()
    assert result.summary == "did it"
    assert len(result.edits) == 1
    assert result.edits[0].path == "a.py"
    assert result.edits[0].content == "x"


async def test_worker_fallback_on_garbage(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = FakeClient(settings, ["total garbage"])
    task = TaskModel(id="1", description="do thing")
    worker = WorkerAgent(task=task, client=client, settings=settings)
    result = await worker.run()
    assert result.edits == []
    assert "parse" in result.summary.lower()


async def test_worker_prompt_includes_file_contents(tmp_path):
    (tmp_path / "a.py").write_text("SECRET_MARKER = 1")
    settings = Settings(workspace=tmp_path)
    client = FakeClient(
        settings, ['{"summary": "ok", "edits": [{"path": "a.py", "content": "x"}]}']
    )
    task = TaskModel(id="1", description="do thing", files=["a.py"])
    worker = WorkerAgent(task=task, client=client, settings=settings)
    await worker.run()
    assert "SECRET_MARKER" in client.prompts[0]


async def test_worker_skips_missing_files(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = FakeClient(settings, ['{"summary": "ok", "edits": []}'])
    task = TaskModel(id="1", description="do thing", files=["missing.py"])
    worker = WorkerAgent(task=task, client=client, settings=settings)
    result = await worker.run()
    assert result.edits == []
    assert "Current file contents" not in client.prompts[0]


# ----------------------------------------------------------------- tester


async def test_tester_runs_custom_command(tmp_path):
    settings = Settings(
        workspace=tmp_path, test_command=f'{sys.executable} -c "print(\'ok\')"'
    )
    client = FakeClient(settings, ['{"passed": true, "summary": "ok", "failures": []}'])
    tester = TesterAgent(client=client, settings=settings)
    result = await tester.run("goal", [])
    assert result.passed is True
    assert result.summary == "ok"
    assert result.failures == []


async def test_tester_no_runner_found(tmp_path):
    settings = Settings(
        workspace=tmp_path, test_command="definitely-not-a-real-command-xyz"
    )
    client = FakeClient(settings, ['{"passed": true, "summary": "ok", "failures": []}'])
    tester = TesterAgent(client=client, settings=settings)
    result = await tester.run("goal", [])
    assert result.passed is True
    assert "No test runner found." in client.prompts[0]


async def test_tester_fallback_heuristic(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = FakeClient(settings, ["Tests passed nicely"])
    tester = TesterAgent(client=client, settings=settings)
    result = await tester.run("goal", [])
    assert result.passed is True
    assert result.summary == "Tests passed nicely"


async def test_tester_parses_failures(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = FakeClient(
        settings, ['{"passed": false, "summary": "2 failed", "failures": ["f1", "f2"]}']
    )
    tester = TesterAgent(client=client, settings=settings)
    result = await tester.run("goal", [])
    assert result.passed is False
    assert result.failures == ["f1", "f2"]
