import asyncio

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def _make_orchestrator(max_cycles: int = 0) -> Orchestrator:
    settings = Settings(max_cycles=max_cycles)
    client = LLMClient(settings=settings)
    return Orchestrator(goal="test goal", client=client)


def _plan_with(tasks: list[TaskModel]) -> Plan:
    return Plan(tasks=tasks, rationale="rationale")


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_no_plan_is_done():
    orch = _make_orchestrator()
    assert orch._is_done() is True


def test_orchestrator_no_tasks_is_done():
    orch = _make_orchestrator()
    orch.current_plan = _plan_with([])
    orch.tests_passed = True
    assert orch._is_done() is True


def test_orchestrator_max_cycles_halts():
    orch = _make_orchestrator(max_cycles=2)
    orch.cycles = 2
    orch.current_plan = _plan_with([TaskModel(id="1", description="x", status="pending")])
    orch.tests_passed = False
    assert orch._is_done() is True


def test_orchestrator_all_complete_and_tests_pass_is_done():
    orch = _make_orchestrator()
    orch.current_plan = _plan_with(
        [TaskModel(id="1", description="x", status="completed")]
    )
    orch.tests_passed = True
    assert orch._is_done() is True


def test_orchestrator_failed_task_not_done():
    orch = _make_orchestrator()
    orch.current_plan = _plan_with(
        [TaskModel(id="1", description="x", status="failed")]
    )
    orch.tests_passed = False
    assert orch._is_done() is False


def test_orchestrator_complete_but_tests_fail_not_done():
    orch = _make_orchestrator()
    orch.current_plan = _plan_with(
        [TaskModel(id="1", description="x", status="completed")]
    )
    orch.tests_passed = False
    assert orch._is_done() is False


def test_orchestrator_pending_tasks_not_done():
    orch = _make_orchestrator()
    orch.current_plan = _plan_with(
        [TaskModel(id="1", description="x", status="pending")]
    )
    orch.tests_passed = False
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_ollama_provider_routed_to_ollama_client(monkeypatch):
    from furrow.config import Provider

    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)

    captured = {}

    async def fake_complete_ollama(prompt, system, model, max_tokens):
        captured["called"] = True
        return "ollama response"

    monkeypatch.setattr(client, "_complete_ollama", fake_complete_ollama)
    result = await client.complete("hi")
    assert result == "ollama response"
    assert captured.get("called") is True


def test_get_tasks_tracks_current_plan():
    orch = _make_orchestrator()
    tasks = [TaskModel(id="1", description="x")]
    orch.current_plan = _plan_with(tasks)
    assert orch._get_tasks() == tasks


@pytest.mark.asyncio
async def test_max_parallel_tasks_enforced(monkeypatch):
    orch = _make_orchestrator()
    orch.client.settings.max_parallel_tasks = 2
    semaphore = asyncio.Semaphore(orch.client.settings.max_parallel_tasks)

    current = 0
    peak = 0

    async def fake_run(self_task):
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.05)
        current -= 1
        return "done"

    monkeypatch.setattr(WorkerAgent, "run", fake_run)
    tasks = [
        orch._run_worker(semaphore, TaskModel(id=str(i), description="d"), orch.client)
        for i in range(5)
    ]
    results = await asyncio.gather(*tasks)

    assert results == ["done"] * 5
    assert peak == 2  # concurrency was capped at max_parallel_tasks

