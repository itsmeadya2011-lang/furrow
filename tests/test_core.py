from __future__ import annotations

import pytest
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


async def test_orchestrator_runs_multiple_cycles_on_test_failure(monkeypatch):
    def fake_plan(self, goal: str) -> Plan:
        if goal.startswith("Fix failing tests"):
            return Plan(tasks=[TaskModel(id="2", description="fix it")], rationale="fix")
        return Plan(tasks=[TaskModel(id="1", description="do it")], rationale="ok")

    def fake_worker(self) -> str:
        return f"completed {self.task.id}"

    def fake_test(self, goal: str, tasks) -> TestResult:
        if goal.startswith("Fix failing tests"):
            return TestResult(passed=True, summary="all good", failures=[])
        return TestResult(passed=False, summary="tests failed", failures=["boom"])

    monkeypatch.setattr(PlannerAgent, "plan", fake_plan)
    monkeypatch.setattr(WorkerAgent, "run", fake_worker)
    monkeypatch.setattr(TesterAgent, "run", fake_test)

    orch = Orchestrator(goal="build a feature", client=LLMClient())
    await orch.run()

    assert orch.cycles > 1
    assert orch.test_passed is True
    assert all(t.status == "completed" for t in orch.tasks)
    assert {t.id for t in orch.tasks} == {"1", "2"}


async def test_orchestrator_retries_failed_task(monkeypatch):
    calls = {"n": 0}

    def fake_plan(self, goal: str) -> Plan:
        return Plan(tasks=[TaskModel(id="1", description="do it")], rationale="ok")

    def fake_worker(self) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom on first attempt")
        return "ok on retry"

    def fake_test(self, goal: str, tasks) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr(PlannerAgent, "plan", fake_plan)
    monkeypatch.setattr(WorkerAgent, "run", fake_worker)
    monkeypatch.setattr(TesterAgent, "run", fake_test)

    orch = Orchestrator(goal="build a feature", client=LLMClient())
    await orch.run()

    assert orch.cycles > 1
    assert calls["n"] >= 2
    assert orch.tasks[0].status == "completed"


async def test_orchestrator_stops_at_max_cycles(monkeypatch):
    def fake_plan(self, goal: str) -> Plan:
        return Plan(tasks=[TaskModel(id="1", description="do it")], rationale="ok")

    def fake_worker(self) -> str:
        return "done"

    def fake_test(self, goal: str, tasks) -> TestResult:
        return TestResult(passed=False, summary="always failing", failures=["x"])

    monkeypatch.setattr(PlannerAgent, "plan", fake_plan)
    monkeypatch.setattr(WorkerAgent, "run", fake_worker)
    monkeypatch.setattr(TesterAgent, "run", fake_test)

    orch = Orchestrator(goal="build a feature", client=LLMClient(), max_cycles=3)
    await orch.run()

    assert orch.cycles == 3
    assert orch.test_passed is False


async def test_orchestrator_stops_when_all_complete(monkeypatch):
    def fake_plan(self, goal: str) -> Plan:
        return Plan(tasks=[TaskModel(id="1", description="do it")], rationale="ok")

    def fake_worker(self) -> str:
        return "done"

    def fake_test(self, goal: str, tasks) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr(PlannerAgent, "plan", fake_plan)
    monkeypatch.setattr(WorkerAgent, "run", fake_worker)
    monkeypatch.setattr(TesterAgent, "run", fake_test)

    orch = Orchestrator(goal="build a feature", client=LLMClient())
    await orch.run()

    assert orch.cycles >= 1
    assert orch.test_passed is True
    assert all(t.status == "completed" for t in orch.tasks)
