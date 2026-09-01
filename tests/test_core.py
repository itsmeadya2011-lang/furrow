import asyncio
from unittest.mock import AsyncMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks_returns_plan_tasks():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.plan = plan
    orchestrator.test_result = None
    tasks = orchestrator._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].description == "do thing"


def test_orchestrator_is_done_with_empty_tasks():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.plan = None
    orchestrator.test_result = None
    orchestrator.cycles = 0
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_completed_with_passing_tests():
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="completed")],
        rationale="ok",
    )
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.plan = plan
    orchestrator.test_result = TestResult(passed=True, summary="ok", failures=[])
    orchestrator.cycles = 1
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_completed_without_tests():
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="completed")],
        rationale="ok",
    )
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.plan = plan
    orchestrator.test_result = None
    orchestrator.cycles = 1
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_failed_tasks():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="do thing", status="completed"),
            TaskModel(id="2", description="other thing", status="failed"),
        ],
        rationale="ok",
    )
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.plan = plan
    orchestrator.test_result = None
    orchestrator.cycles = 1
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_respects_max_cycles():
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="completed")],
        rationale="ok",
    )
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.plan = plan
    orchestrator.test_result = TestResult(passed=True, summary="ok", failures=[])
    orchestrator.cycles = 5
    settings = Settings(max_cycles=3)
    orchestrator.client = LLMClient(settings=settings)
    assert orchestrator._is_done() is True


def test_worker_apply_writes_and_summarize():
    client = LLMClient()
    client.write_file = AsyncMock()
    worker = WorkerAgent.__new__(WorkerAgent)
    worker.client = client
    response = (
        "WRITE: src/main.py\n"
        "print('hello')\n"
        "---END---\n"
        "WRITE: tests/test_main.py\n"
        "def test_main():\n"
        "    pass\n"
        "---END---\n"
        "Summary: created files."
    )
    result = asyncio.run(worker._apply_writes_and_summarize(response))
    assert "src/main.py" in result
    assert "tests/test_main.py" in result
    assert "Summary: created files." in result


def test_worker_apply_writes_handles_no_writes():
    client = LLMClient()
    client.write_file = AsyncMock()
    worker = WorkerAgent.__new__(WorkerAgent)
    worker.client = client
    response = "Just a plain summary without file writes."
    result = asyncio.run(worker._apply_writes_and_summarize(response))
    assert result == response


def test_llm_client_ollama_property():
    client = LLMClient(settings=Settings(provider="ollama"))
    assert client.ollama is not None
