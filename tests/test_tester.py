"""Unit tests for the tester verdict helpers and its error handling.

Every ``TesterAgent.run`` test monkeypatches ``_run_tests``: the real one would
spawn ``pytest`` and recursively run this very suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from furrow.agents.tester import (
    RunnerTimeout,
    _as_bool,
    _fallback_verdict,
    _verdict,
)
from furrow.agents.tester import (
    TesterAgent as _TesterAgent,  # aliased: pytest collects Test* names
)
from furrow.config import Settings, TaskModel
from furrow.config import TestResult as _TestResult

# --------------------------------------------------------------------------- #
# _as_bool
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        ("  yes  ", True),
        ("pass", True),
        ("passed", True),
        ("success", True),
        ("ok", True),
        ("false", False),
        ("no", False),
        ("fail", False),
        ("failed", False),
        ("failure", False),
        ("error", False),
        ("maybe", None),
        ("", None),
        ("y", None),
        (None, None),
        (1, None),
        (0, None),
        ([], None),
        ({}, None),
    ],
)
def test_as_bool(value, expected):
    assert _as_bool(value) is expected


# --------------------------------------------------------------------------- #
# _verdict
# --------------------------------------------------------------------------- #


def test_verdict_from_a_well_formed_dict():
    result = _verdict({"passed": True, "summary": "all good", "failures": []})
    assert isinstance(result, _TestResult)
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


def test_verdict_stringifies_failures_and_defaults_the_summary():
    result = _verdict({"passed": "no", "failures": [1, "boom"]})
    assert result is not None
    assert result.passed is False
    assert result.summary == "Tester returned no summary."
    assert result.failures == ["1", "boom"]


def test_verdict_missing_failures_key_defaults_to_empty():
    result = _verdict({"passed": True})
    assert result is not None
    assert result.failures == []


def test_verdict_without_passed_is_unusable():
    assert _verdict({"summary": "everything is fine"}) is None


def test_verdict_with_uninterpretable_passed_is_unusable():
    assert _verdict({"passed": "maybe"}) is None


def test_verdict_with_non_list_failures_is_unusable():
    assert _verdict({"passed": True, "failures": "boom"}) is None


# --------------------------------------------------------------------------- #
# _fallback_verdict
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "response", ["PASS", "passed", "ok", "Okay", "success", "GREEN", "  **pass**  "]
)
def test_fallback_verdict_recognises_pass_markers(response):
    passed, _ = _fallback_verdict(response)
    assert passed is True


@pytest.mark.parametrize("response", ["FAIL", "failed", "failure", "error", "RED"])
def test_fallback_verdict_recognises_fail_markers(response):
    passed, _ = _fallback_verdict(response)
    assert passed is False


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("all tests passed", True),
        ("All tests passed.", True),
        ("test passed", True),
        ("Tests failed.", False),
        ("test failed", False),
    ],
)
def test_fallback_verdict_uses_the_tests_regex(response, expected):
    passed, _ = _fallback_verdict(response)
    assert passed is expected


# Regression: the old check was `"passed" in response.lower()`, which is True for
# "not passed". A negated verdict must never be read as a pass.
@pytest.mark.parametrize("response", ["not passed", "NOT PASSED", "did not pass", "unpassed"])
def test_fallback_verdict_rejects_a_negated_pass(response):
    passed, reason = _fallback_verdict(response)
    assert passed is False
    assert "inconclusive" in reason


def test_fallback_verdict_prose_is_inconclusive_and_not_a_pass():
    response = "I looked over the code and it seems reasonable overall, but I have doubts."
    passed, reason = _fallback_verdict(response)
    assert passed is False
    assert "inconclusive" in reason


def test_fallback_verdict_empty_response_is_not_a_pass():
    passed, reason = _fallback_verdict("")
    assert passed is False
    assert "inconclusive" in reason


def test_fallback_verdict_only_looks_at_the_first_non_empty_line():
    passed, _ = _fallback_verdict("\n\nPASS\nalthough everything else looks broken")
    assert passed is True


# --------------------------------------------------------------------------- #
# TesterAgent.run
# --------------------------------------------------------------------------- #


class _FakeClient:
    def __init__(self, response: str, settings: Settings) -> None:
        self.response = response
        self.settings = settings
        self.prompts: list[str] = []

    async def complete(self, prompt, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.response


def _agent(tmp_path, response: str) -> tuple[_TesterAgent, _FakeClient]:
    settings = Settings(_env_file=None, workspace=Path(tmp_path))
    client = _FakeClient(response, settings)
    return _TesterAgent(client=client, settings=settings), client


def _raise(exc: Exception):
    async def _boom():
        raise exc

    return _boom


async def test_run_reports_a_runner_timeout_as_a_failure(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, '{"passed": true, "summary": "ignored"}')
    monkeypatch.setattr(agent, "_run_tests", _raise(RunnerTimeout("Test runner timed out.")))

    result = await agent.run("goal", [])

    assert isinstance(result, _TestResult)
    assert result.passed is False
    assert "timed out" in result.summary
    assert result.failures == [result.summary]


async def test_run_distinguishes_a_crashed_runner_from_failing_tests(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, '{"passed": true, "summary": "ignored"}')
    monkeypatch.setattr(agent, "_run_tests", _raise(OSError("the disk is on fire")))

    result = await agent.run("goal", [])

    assert result.passed is False
    assert result.summary.startswith("Test runner crashed: ")
    assert "the disk is on fire" in result.summary


async def test_run_uses_the_json_verdict_when_present(tmp_path, monkeypatch):
    agent, client = _agent(
        tmp_path, '```json\n{"passed": true, "summary": "3 passed", "failures": []}\n```'
    )

    async def _run():
        return "3 passed in 0.01s"

    monkeypatch.setattr(agent, "_run_tests", _run)

    result = await agent.run("goal", [TaskModel(id="1", description="d", files=["a.py"])])

    assert result.passed is True
    assert result.summary == "3 passed"
    assert result.failures == []
    # The goal, the files under test and the raw runner output all reach the model.
    assert "goal" in client.prompts[0]
    assert "a.py" in client.prompts[0]
    assert "3 passed in 0.01s" in client.prompts[0]


async def test_run_falls_back_to_the_prose_verdict(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, "All tests passed.")

    async def _run():
        return "no idea"

    monkeypatch.setattr(agent, "_run_tests", _run)

    result = await agent.run("goal", [])

    assert result.passed is True
    assert "All tests passed." in result.summary
    assert result.failures == []


async def test_run_treats_unusable_json_as_no_json(tmp_path, monkeypatch):
    agent, _ = _agent(tmp_path, '{"summary": "no verdict here"}')

    async def _run():
        return "boom"

    monkeypatch.setattr(agent, "_run_tests", _run)

    result = await agent.run("goal", [])

    # The dict has no usable "passed", so the prose fallback decides: not a pass.
    assert result.passed is False
    assert "no verdict here" in result.summary
