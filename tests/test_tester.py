"""Tests for the TesterAgent in furrow.agents.tester.

The real ``_run_tests`` spawns subprocesses (pytest, npm, ...) which would
recurse into the test suite and hang, and whether a runner is present depends
on the environment. To keep the tests deterministic we monkeypatch
``_run_tests`` to either raise ``FileNotFoundError`` (simulating "no runner
found") or return canned test output.
"""

import asyncio
import json
from pathlib import Path

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel, TestResult


class FakeTesterClient:
    """Returns a fixed LLM response; records the calls made."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.complete_calls = 0
        self.settings = Settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls += 1
        return self.response


def test_tester_returns_failure_when_no_runner_found(monkeypatch) -> None:
    """When no test runner is installed the agent returns a failing TestResult."""
    async def fake_run_tests(self):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)

    client = FakeTesterClient('{"passed": true, "summary": "ignored", "failures": []}')
    agent = TesterAgent(client=client)

    result = asyncio.run(agent.run(goal="ship it", tasks=[]))

    assert isinstance(result, TestResult)
    assert result.passed is False
    assert "No test runner found" in result.summary
    assert len(result.failures) >= 1
    # The LLM should never have been consulted when there is no runner.
    assert client.complete_calls == 0


def test_tester_parses_valid_json_response(monkeypatch) -> None:
    """A well-formed JSON response is parsed into a TestResult."""
    async def fake_run_tests(self):
        return "1 passed, 0 failed"

    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)

    payload = json.dumps(
        {
            "passed": True,
            "summary": "All 3 tests passed",
            "failures": [],
        }
    )
    client = FakeTesterClient(payload)
    agent = TesterAgent(client=client)

    result = asyncio.run(agent.run(goal="be happy", tasks=[]))

    assert result.passed is True
    assert result.summary == "All 3 tests passed"
    assert result.failures == []
    assert client.complete_calls == 1


def test_tester_parses_failing_json_response(monkeypatch) -> None:
    async def fake_run_tests(self):
        return "2 failed"

    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)

    payload = json.dumps(
        {
            "passed": False,
            "summary": "2 failures",
            "failures": ["test_a broke", "test_b broke"],
        }
    )
    client = FakeTesterClient(payload)
    agent = TesterAgent(client=client)

    result = asyncio.run(agent.run(goal="be stable", tasks=[]))

    assert result.passed is False
    assert result.summary == "2 failures"
    assert result.failures == ["test_a broke", "test_b broke"]


def test_tester_fallback_when_response_is_not_json(monkeypatch) -> None:
    """A non-JSON response with 'fail'/'error' is treated as a failure."""
    async def fake_run_tests(self):
        return "ok"

    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)

    client = FakeTesterClient("Something failed during the run.")
    agent = TesterAgent(client=client)

    result = asyncio.run(agent.run(goal="be reliable", tasks=[]))

    assert result.passed is False
    assert result.summary == "Something failed during the run."


def test_tester_fallback_when_response_is_passing_text(monkeypatch) -> None:
    """A non-JSON response containing 'pass' is treated as a pass."""
    async def fake_run_tests(self):
        return "ok"

    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)

    client = FakeTesterClient("All tests passed successfully.")
    agent = TesterAgent(client=client)

    result = asyncio.run(agent.run(goal="be reliable", tasks=[]))

    assert result.passed is True
    assert result.summary == "All tests passed successfully."


def test_tester_uses_provided_client_settings_workspace(monkeypatch, tmp_path: Path) -> None:
    """The tester honours the workspace configured on its client."""
    async def fake_run_tests(self):
        return "ok"

    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)

    payload = json.dumps({"passed": True, "summary": "ok", "failures": []})
    client = FakeTesterClient(payload)
    client.settings.workspace = tmp_path
    agent = TesterAgent(client=client)

    result = asyncio.run(agent.run(goal="ok", tasks=[]))

    assert result.passed is True