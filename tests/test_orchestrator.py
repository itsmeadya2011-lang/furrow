import asyncio
from unittest.mock import AsyncMock

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class FakeLLM(LLMClient):
    def __init__(self):
        super().__init__()
        self._calls = 0

    async def complete(self, prompt, system="", model=None):
        self._calls += 1
        # TesterAgent prompts contain "Test output:"; return a passing TestResult.
        if "Test output:" in prompt:
            return '{"passed": true, "summary": "ok", "failures": []}'
        # First planner call returns one task; subsequent planner calls return empty
        # so the orchestrator halts after one cycle of real work.
        if self._calls == 1:
            return '{"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"}'
        return '{"tasks": [], "rationale": "nothing left"}'


@pytest.mark.asyncio
async def test_orchestrator_run_completes_tasks(monkeypatch):
    # Prevent the real tester from spawning a recursive pytest subprocess.
    monkeypatch.setattr(TesterAgent, "_run_tests", AsyncMock(return_value="all tests passed"))
    client = FakeLLM()
    orch = Orchestrator("goal", client=client)
    await orch.run()
    assert len(orch.all_tasks) >= 1
    assert all(t.status == "completed" for t in orch.all_tasks)
    # At least the first cycle ran. If the planner returned a non-empty plan
    # in cycle 1, _is_done returns True and the loop halts after exactly 1 cycle.
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles(monkeypatch):
    # Planner always returns a non-empty plan; the worker fails; only max_cycles stops the loop.
    monkeypatch.setattr(TesterAgent, "_run_tests", AsyncMock(return_value="ok"))
    monkeypatch.setattr(settings, "max_cycles", 3)
    async def fake_complete(self, prompt, system="", model=None):
        # TesterAgent prompts contain "Test output:"; return a passing TestResult for those.
        if "Test output:" in prompt:
            return '{"passed": true, "summary": "ok", "failures": []}'
        return '{"tasks": [{"id": "1", "description": "x"}], "rationale": "ok"}'
    monkeypatch.setattr(LLMClient, "complete", fake_complete)
    async def fake_worker_run(self):
        raise RuntimeError("simulated worker failure")
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent.run", fake_worker_run)
    orch = Orchestrator("goal", client=LLMClient())
    await orch.run()
    assert orch.cycles == 3
