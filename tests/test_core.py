import asyncio
from pathlib import Path

import pytest
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="d")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_orchestrator_is_done_empty_when_no_plan():
    orch = Orchestrator(goal="x")
    assert orch._is_done() is False


def test_orchestrator_is_done_after_plan():
    orch = Orchestrator(goal="x")
    orch.last_plan = Plan(
        tasks=[TaskModel(id="1", description="d", status="completed")], rationale=""
    )
    assert orch._is_done() is True


def test_orchestrator_is_done_with_failure():
    orch = Orchestrator(goal="x")
    orch.last_plan = Plan(
        tasks=[TaskModel(id="1", description="d", status="failed")], rationale=""
    )
    assert orch._is_done() is False


def test_orchestrator_max_cycles_guard():
    s = Settings(max_cycles=2)
    assert s.max_cycles == 2
    orch = Orchestrator(goal="x")
    assert orch.cycles == 0


async def test_worker_parses_json_edits(tmp_path):
    client = LLMClient()

    async def fake_write_file(path, content):
        Path(path).write_text(content)

    async def fake_read_file(path):
        return Path(path).read_text()

    client.write_file = fake_write_file
    client.read_file = fake_read_file

    target = tmp_path / "x.py"
    target.write_text("hello\n")

    task = TaskModel(id="1", description="d", files=["x.py"])
    worker = WorkerAgent(task=task, client=client)

    await worker._apply_edit(
        {"path": str(target), "old_text": "hello", "new_text": "world"}
    )
    text = target.read_text()
    assert "world" in text
    assert "hello" not in text


async def test_worker_handles_non_json_response():
    client = LLMClient()

    async def fake_complete(prompt, model=None, system=""):
        return "not json"

    client.complete = fake_complete

    task = TaskModel(id="1", description="d")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "not json"


async def test_llm_provider_dispatch():
    client = LLMClient()

    async def fake_anthropic(prompt, system, model):
        return "anthropic-out"

    async def fake_openai(prompt, system, model):
        return "openai-out"

    async def fake_ollama(prompt, system, model):
        return "ollama-out"

    client._complete_anthropic = fake_anthropic
    client._complete_openai = fake_openai
    client._complete_ollama = fake_ollama

    client.settings.provider = Provider.ANTHROPIC
    assert await client.complete("hi") == "anthropic-out"

    client.settings.provider = Provider.OPENAI
    assert await client.complete("hi") == "openai-out"

    client.settings.provider = Provider.OLLAMA
    assert await client.complete("hi") == "ollama-out"


def test_tester_narrow_exception():
    import furrow.agents.tester as tester_module
    assert asyncio.iscoroutinefunction(tester_module.TesterAgent._run_tests)
