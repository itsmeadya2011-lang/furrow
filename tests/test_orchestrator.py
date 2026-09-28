from __future__ import annotations

import pytest

from furrow.agents.tester import TesterAgent
from furrow.core.orchestrator import Orchestrator
from tests._fakes import (
    FakeLLMClient,
    FakeSettings,
    fake_run_tests_ok,
    fake_run_tests_raise,
    make_plan_json,
    make_test_result,
)


def _orch(
    plans,
    workers,
    tests,
    *,
    max_cycles: int = 0,
    max_parallel: int = 5,
    worker_exc=None,
) -> Orchestrator:
    settings = FakeSettings()
    settings.max_cycles = max_cycles
    settings.max_parallel_tasks = max_parallel
    client = FakeLLMClient(
        plans=plans,
        workers=workers,
        test_results=tests,
        settings=settings,
        worker_exc=worker_exc,
    )
    return Orchestrator(goal="do a thing", client=client, settings=settings)


@pytest.mark.asyncio
async def test_completes_when_tests_pass(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests_ok)
    orch = _orch(
        [make_plan_json()], ["summary"], [make_test_result(passed=True)]
    )
    await orch.run()
    assert orch.cycles == 1
    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "completed"
    assert orch.last_test_passed is True
    assert orch.last_test_result is not None


@pytest.mark.asyncio
async def test_retries_on_test_failure(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests_ok)
    orch = _orch(
        [make_plan_json(), make_plan_json()],
        ["s1", "s2"],
        [
            make_test_result(passed=False, summary="bad", failures=["fail X"]),
            make_test_result(passed=True),
        ],
    )
    await orch.run()
    assert orch.cycles == 2
    assert orch.last_test_passed is True
    assert "Fix failing tests" in orch.goal


@pytest.mark.asyncio
async def test_stops_at_max_cycles(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests_raise)
    orch = _orch(
        [make_plan_json(), make_plan_json()],
        ["s1", "s2"],
        [make_test_result(passed=False)],
        max_cycles=1,
    )
    await orch.run()
    assert orch.cycles == 1
    assert orch.last_test_passed is False


@pytest.mark.asyncio
async def test_no_tasks_completes(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests_ok)
    orch = _orch(
        ['{"tasks": [], "rationale": "already done"}'],
        [],
        [make_test_result(passed=True)],
    )
    await orch.run()
    assert orch.cycles == 1
    assert orch.tasks == []
    assert orch.last_test_result is None


@pytest.mark.asyncio
async def test_failed_task_keeps_looping_until_max_cycles(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests_ok)
    orch = _orch(
        [make_plan_json(), make_plan_json()],
        None,
        [make_test_result(passed=True), make_test_result(passed=True)],
        max_cycles=2,
        worker_exc=RuntimeError("worker crashed"),
    )
    await orch.run()
    assert orch.cycles == 2
    assert orch.tasks[0].status == "failed"
    assert orch.last_test_passed is True
