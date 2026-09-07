import pytest
from unittest.mock import MagicMock, AsyncMock

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


@pytest.mark.asyncio
async def test_test_result_success():
    client = MagicMock()
    client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="test output")
    result = await tester.run("test goal", [])
    assert result.passed is True
    assert result.summary == "ok"


@pytest.mark.asyncio
async def test_test_result_retry_on_bad_json():
    client = MagicMock()
    client.complete = AsyncMock(side_effect=[
        "bad json",
        '{"passed": true, "summary": "ok", "failures": []}',
    ])
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="test output")
    result = await tester.run("test goal", [])
    assert result.passed is True
    assert client.complete.call_count == 2


@pytest.mark.asyncio
async def test_test_result_heuristic_fallback():
    client = MagicMock()
    client.complete = AsyncMock(return_value="all tests passed")
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="test output")
    result = await tester.run("test goal", [])
    assert result.passed is True
    assert "all tests passed" in result.summary
