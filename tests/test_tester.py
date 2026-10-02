"""Tests for the TesterAgent."""

import json

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


class FakeLLMClient:
    """Fake LLM client that returns scripted JSON responses."""

    def __init__(self, json_response: str) -> None:
        self._json_response = json_response
        self._test_output = ""
        self.settings = type(
            "S",
            (),
            {
                "workspace": ".",
                "tester_model": "fake-tester",
                "test_timeout": 30,
            },
        )()

    async def list_files(self, directory):
        return []

    async def complete_json(
        self, prompt, system="", model=None, temperature=0.7, max_tokens=4096, retries=2
    ):
        return json.loads(self._json_response)


@pytest.fixture
def passing_test_result():
    return json.dumps(
        {
            "passed": True,
            "summary": "All 12 tests passed",
            "failures": [],
        }
    )


@pytest.fixture
def failing_test_result():
    return json.dumps(
        {
            "passed": False,
            "summary": "3 tests failed",
            "failures": [
                "test_auth.py::test_login: AssertionError",
                "test_auth.py::test_logout: KeyError: 'user'",
            ],
        }
    )


@pytest.fixture
def invalid_json():
    return "not json"


@pytest.fixture
def simple_tasks():
    return [TaskModel(id="1", description="Add auth")]


@pytest.fixture
def tester_with_fakes(passing_test_result):
    return TesterAgent(client=FakeLLMClient(passing_test_result))


def test_tester_init():
    """TesterAgent initializes with a client."""
    tester = TesterAgent()
    assert tester.client is not None


@pytest.mark.asyncio
async def test_tester_passing_result(tester_with_fakes, simple_tasks):
    """Tester returns passing result when LLM says pass."""
    tester = tester_with_fakes

    async def passing_output():
        return "12 passed in 0.5s"

    tester._run_tests = passing_output
    result = await tester.run("Implement auth", simple_tasks)
    assert result.passed is True
    assert result.summary == "All 12 tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_failing_result(failing_test_result, simple_tasks):
    """Tester returns failing result with failure details."""
    tester = TesterAgent(client=FakeLLMClient(failing_test_result))

    async def failing_output():
        return "3 failed, 9 passed"

    tester._run_tests = failing_output
    result = await tester.run("Fix failing auth tests", simple_tasks)
    assert result.passed is False
    assert len(result.failures) == 2
    assert "test_login" in result.failures[0]


@pytest.mark.asyncio
async def test_tester_heuristic_eval_on_parse_error(invalid_json, simple_tasks):
    """Tester falls back to heuristic evaluation when LLM parsing fails."""
    tester = TesterAgent(client=FakeLLMClient(invalid_json))

    async def failing_output():
        return "3 failed, 9 passed"

    tester._run_tests = failing_output
    result = await tester.run("Some goal", simple_tasks)
    assert isinstance(result, TestResult)
    assert result.passed is False


@pytest.mark.asyncio
async def test_tester_no_test_runner():
    """Tester handles missing test runner gracefully."""
    tester = TesterAgent(
        client=FakeLLMClient(json.dumps({"passed": True, "summary": "ok", "failures": []}))
    )

    async def no_runner():
        return "No test runner found."

    tester._run_tests = no_runner
    result = await tester.run("Some goal", [])
    assert result.passed is True


@pytest.mark.asyncio
async def test_tester_handles_test_errors(simple_tasks):
    """Tester handles exceptions during test execution."""
    tester = TesterAgent(
        client=FakeLLMClient(
            json.dumps({"passed": False, "summary": "error", "failures": ["test error"]})
        )
    )

    async def raise_error():
        raise RuntimeError("Test runner crashed")

    tester._run_tests = raise_error
    result = await tester.run("Some goal", simple_tasks)
    assert result.passed is False
    assert isinstance(result, TestResult)


@pytest.mark.asyncio
async def test_tester_timeout_message():
    """Tester returns timeout message when tests exceed timeout."""
    tester = TesterAgent(
        client=FakeLLMClient(json.dumps({"passed": True, "summary": "ok", "failures": []}))
    )

    async def timeout_msg():
        return "[test runner 'pytest' timed out after 30s]"

    tester._run_tests = timeout_msg
    result = await tester.run("Some goal", [])
    assert isinstance(result, TestResult)


@pytest.mark.asyncio
async def test_tester_heuristic_detects_pass():
    """Heuristic evaluation detects passing tests."""
    tester = TesterAgent(
        client=FakeLLMClient(json.dumps({"passed": False, "summary": "err", "failures": []}))
    )

    async def passing_output():
        return "12 passed in 0.5s"

    tester._run_tests = passing_output
    result = await tester.run("Some goal", [])
    assert result.passed is True


@pytest.mark.asyncio
async def test_tester_heuristic_detects_fail():
    """Heuristic evaluation detects failing tests."""
    tester = TesterAgent(
        client=FakeLLMClient(json.dumps({"passed": True, "summary": "ok", "failures": []}))
    )

    async def failing_output():
        return "3 failed, 9 passed"

    tester._run_tests = failing_output
    result = await tester.run("Some goal", [])
    assert result.passed is False
