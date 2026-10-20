"""Tests for the tester agent, in particular the false-pass regression."""

from __future__ import annotations

from furrow.agents.tester import TesterAgent
from furrow.config import Settings
from tests.helpers import FakeLLMClient, make_test_result


async def test_tester_does_not_call_prose_that_mentions_passing_a_pass(settings: Settings) -> None:
    client = FakeLLMClient(settings, tests=["All tests passed"])

    result = await TesterAgent(client=client).run("goal", [])

    assert result.passed is False


async def test_tester_does_not_call_truncated_json_a_pass(settings: Settings) -> None:
    client = FakeLLMClient(settings, tests=['{"passed": false, "summary": "1 failed, truncat'])

    result = await TesterAgent(client=client).run("goal", [])

    assert result.passed is False


async def test_tester_does_not_call_a_pass_when_the_payload_is_ambiguous(
    settings: Settings,
) -> None:
    client = FakeLLMClient(settings, tests=["3 passed, 0 failed, everything is green"])

    result = await TesterAgent(client=client).run("goal", [])

    assert result.passed is False


async def test_tester_reports_an_explicit_failure(settings: Settings) -> None:
    payload = make_test_result(False, "boom", ["tests/test_x.py::test_y"])
    client = FakeLLMClient(settings, tests=[payload])

    result = await TesterAgent(client=client).run("goal", [])

    assert result.passed is False
    assert result.failures == ["tests/test_x.py::test_y"]
    assert result.summary == "boom"


async def test_tester_reports_an_explicit_pass(settings: Settings) -> None:
    client = FakeLLMClient(settings, tests=[make_test_result(True, "all good")])

    result = await TesterAgent(client=client).run("goal", [])

    assert result.passed is True
    assert result.failures == []
    assert result.summary == "all good"


async def test_tester_accepts_a_minimal_pass_payload(settings: Settings) -> None:
    client = FakeLLMClient(settings, tests=['{"passed": true}'])

    result = await TesterAgent(client=client).run("goal", [])

    assert result.passed is True


async def test_tester_reports_a_failure_when_the_runner_explodes(
    settings: Settings, monkeypatch
) -> None:
    async def _boom(self: TesterAgent) -> str:
        raise RuntimeError("no test runner")

    monkeypatch.setattr(TesterAgent, "_run_tests", _boom)
    client = FakeLLMClient(settings, tests=[make_test_result(True)])

    result = await TesterAgent(client=client).run("goal", [])

    assert result.passed is False
    assert client.calls == []


async def test_tester_asks_the_model_with_the_goal_and_the_test_output(settings: Settings) -> None:
    client = FakeLLMClient(settings, tests=[make_test_result(True)])

    await TesterAgent(client=client).run("build a parser", [])

    call = client.calls[-1]
    assert "build a parser" in call["prompt"]
    assert "stubbed test output" in call["prompt"]
    assert call["model"] == settings.tester_model
