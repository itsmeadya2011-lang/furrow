from __future__ import annotations

import json

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator


def _settings(**overrides):
    return Settings(**overrides)


class FakeLLM:
    """Duck-typed stand-in for LLMClient to drive the orchestrator deterministically."""

    def __init__(self, settings: Settings, *, fail_tests: bool = False) -> None:
        self.settings = settings
        self.fail_tests = fail_tests

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if "rationale" in prompt and "tasks" in prompt:
            return json.dumps(
                {
                    "tasks": [
                        {"id": "1", "description": "stub task", "files": [], "dependencies": []},
                        {"id": "2", "description": "stub task 2", "files": [], "dependencies": []},
                    ],
                    "rationale": "stub plan",
                }
            )
        if "Test output" in prompt:
            if self.fail_tests:
                return json.dumps(
                    {"passed": False, "summary": "tests failed", "failures": ["something broke"]}
                )
            return json.dumps({"passed": True, "summary": "all good", "failures": []})
        return "Stub implementation complete."


@pytest.fixture
def fake_tests(monkeypatch):
    async def _fake_run_tests(self):
        return "all tests passed"

    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)


async def test_orchestrator_completes_when_tests_pass(fake_tests):
    settings = _settings(max_cycles=1, max_parallel_tasks=2)
    orch = Orchestrator(goal="demo", client=FakeLLM(settings))
    await orch.run()
    assert orch.cycles == 1
    assert all(t.status == "completed" for t in orch._tasks)
    assert orch._test_passed is True


async def test_orchestrator_respects_max_cycles(fake_tests):
    settings = _settings(max_cycles=2, max_parallel_tasks=2)
    orch = Orchestrator(goal="demo", client=FakeLLM(settings, fail_tests=True))
    await orch.run()
    assert orch.cycles == 2
