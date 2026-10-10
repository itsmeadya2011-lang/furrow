from unittest.mock import AsyncMock, Mock

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TestResult


def test_tester_init():
    tester = TesterAgent()
    assert tester.client is not None


def test_tester_init_with_client():
    mock_client = Mock()
    tester = TesterAgent(client=mock_client)
    assert tester.client == mock_client


@pytest.mark.asyncio
async def test_tester_run_tests_not_found():
    """When no test runner is found, should return failed TestResult."""
    settings = Settings(_env_file=None)
    client = Mock()
    client.settings = settings
    client.complete = AsyncMock(return_value='{"passed": false, "summary": "no tests", "failures": ["no runner"]}')

    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="No test runner found.")
    result = await tester.run("test goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is False


@pytest.mark.asyncio
async def test_tester_parse_json_response():
    """Tester should parse valid JSON response from LLM."""
    settings = Settings(_env_file=None)
    client = Mock()
    client.settings = settings
    client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "All good", "failures": []}'
    )

    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="All tests passed")
    result = await tester.run("test goal", [])
    assert result.passed is True
    assert result.summary == "All good"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_parse_json_in_code_block():
    """Tester should extract JSON from markdown code blocks."""
    settings = Settings(_env_file=None)
    client = Mock()
    client.settings = settings
    client.complete = AsyncMock(
        return_value='```json\n{"passed": false, "summary": "broken", "failures": ["test_x"]}\n```'
    )

    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="Some test output")
    result = await tester.run("test goal", [])
    assert result.passed is False
    assert "test_x" in result.failures


@pytest.mark.asyncio
async def test_tester_fallback_when_no_json():
    """When LLM returns non-JSON, should infer from test output."""
    settings = Settings(_env_file=None)
    client = Mock()
    client.settings = settings
    client.complete = AsyncMock(return_value="Something went wrong, tests failed")

    # Patch _run_tests to return output with failure indicators
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="FAILED tests/test_x.py::test_fail")

    result = await tester.run("test goal", [])
    assert result.passed is False


@pytest.mark.asyncio
async def test_tester_fallback_success():
    """When LLM returns non-JSON but tests passed, should infer success."""
    settings = Settings(_env_file=None)
    client = Mock()
    client.settings = settings
    client.complete = AsyncMock(return_value="Looks good I guess")

    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="All tests passed in 1.23s")

    result = await tester.run("test goal", [])
    assert result.passed is True


def test_infer_passed_from_output():
    """Direct test of the pass/fail inference logic."""
    tester = TesterAgent()
    assert tester._infer_passed_from_output("All tests passed") is True
    assert tester._infer_passed_from_output("5 passed, 0 failed") is True
    assert tester._infer_passed_from_output("FAILED tests/test.py::test_x") is False
    assert tester._infer_passed_from_output("Error: something broke") is False
    assert tester._infer_passed_from_output("Traceback (most recent call last):") is False
    assert tester._infer_passed_from_output("") is False


def test_extract_json_plain():
    """Test JSON extraction from plain text."""
    tester = TesterAgent()
    data = tester._extract_json('{"passed": true, "summary": "ok", "failures": []}')
    assert data["passed"] is True


def test_extract_json_code_block():
    """Test JSON extraction from markdown code block."""
    tester = TesterAgent()
    text = 'Here is the result:\n```json\n{"passed": false, "summary": "bad", "failures": ["x"]}\n```\nDone.'
    data = tester._extract_json(text)
    assert data["passed"] is False
    assert data["summary"] == "bad"


def test_extract_json_no_json_raises():
    """Test that missing JSON raises JSONDecodeError."""
    import json
    tester = TesterAgent()
    with pytest.raises(json.JSONDecodeError):
        tester._extract_json("no json here at all")