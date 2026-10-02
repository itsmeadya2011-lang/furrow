"""Tests for furrow.agents.tester: TesterAgent parsing, keyword fallback, and test running."""

import json
from unittest.mock import AsyncMock

from furrow.agents.tester import TesterAgent
from furrow.config import Settings


class MockLLMClient:
    def __init__(self, settings: Settings, response: str) -> None:
        self.settings = settings
        self.complete = AsyncMock(return_value=response)


def make_settings() -> Settings:
    return Settings(provider="anthropic", anthropic_api_key="sk-test")


def make_client(response: str) -> "MockLLMClient":
    return MockLLMClient(settings=make_settings(), response=response)


def make_tester(response: str = "") -> "TesterAgent":
    return TesterAgent(client=make_client(response))


def test_parse_test_result_valid_json():
    tester = make_tester()
    response = json.dumps({"passed": True, "summary": "all good", "failures": []})
    result = tester._parse_test_result_response(response)
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


def test_parse_test_result_valid_json_with_failures():
    tester = make_tester()
    response = json.dumps({"passed": False, "summary": "2 failed", "failures": ["a", "b"]})
    result = tester._parse_test_result_response(response)
    assert result.passed is False
    assert result.failures == ["a", "b"]


def test_parse_test_result_markdown_code_block():
    tester = make_tester()
    response = '```json\n{"passed": true, "summary": "ok", "failures": []}\n```'
    result = tester._parse_test_result_response(response)
    assert result.passed is True
    assert result.summary == "ok"


def test_parse_test_result_keyword_fallback_failure():
    tester = make_tester()
    result = tester._parse_test_result_response("some tests failed with 1 error")
    assert result.passed is False


def test_parse_test_result_keyword_fallback_pass():
    tester = make_tester()
    result = tester._parse_test_result_response("All tests passed successfully")
    assert result.passed is True


def test_parse_test_result_keyword_fallback_success():
    tester = make_tester()
    result = tester._parse_test_result_response("build succeeded with status OK")
    assert result.passed is True


def test_parse_test_result_not_passed_does_not_match_as_passed():
    tester = make_tester()
    result = tester._parse_test_result_response("status: not_passed")
    assert result.passed is False


def test_parse_test_result_non_json_no_keywords_defaults_false():
    tester = make_tester()
    result = tester._parse_test_result_response("nothing relevant here")
    assert result.passed is False


async def test_tester_run_returns_test_result():
    client = make_client(json.dumps({"passed": True, "summary": "ok", "failures": []}))
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="test output here")
    result = await tester.run("build feature", [])
    assert result.passed is True
    assert result.summary == "ok"
    client.complete.assert_awaited_once()
    tester._run_tests.assert_awaited_once()


async def test_tester_run_passes_test_output_to_llm():
    client = make_client(json.dumps({"passed": False, "summary": "broken", "failures": ["x"]}))
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="FAIL: test_x failed")
    result = await tester.run("goal", [])
    assert result.passed is False
    prompt_sent = client.complete.await_args.args[0]
    assert "FAIL: test_x failed" in prompt_sent


async def test_tester_run_handles_test_runner_exception():
    client = make_client(json.dumps({"passed": True, "summary": "ok", "failures": []}))
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(side_effect=RuntimeError("runner crashed"))
    result = await tester.run("goal", [])
    assert result.passed is False
    assert "runner crashed" in result.summary
    assert "runner crashed" in result.failures[0]
    client.complete.assert_not_awaited()


async def test_run_tests_no_runner_found(monkeypatch):
    monkeypatch.setattr(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        AsyncMock(side_effect=OSError("command not found")),
    )
    tester = make_tester()
    result = await tester._run_tests()
    assert result == "No test runner found."


async def test_run_tests_finds_and_runs_pytest(monkeypatch):
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"5 passed in 0.1s", b""))
    mock_proc.returncode = 0
    mock_proc.kill = AsyncMock()
    mock_proc.wait = AsyncMock()
    create_mock = AsyncMock(return_value=mock_proc)
    monkeypatch.setattr("furrow.agents.tester.asyncio.create_subprocess_exec", create_mock)
    tester = make_tester()
    result = await tester._run_tests()
    assert "5 passed" in result
    assert create_mock.await_count == 1
    cmd = create_mock.await_args.args
    assert cmd[0] == "pytest"

