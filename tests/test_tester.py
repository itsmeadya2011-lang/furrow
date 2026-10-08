from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.complete = AsyncMock()
    return client


@pytest.fixture
def tester(mock_client: MagicMock) -> TesterAgent:
    return TesterAgent(client=mock_client)


async def test_tester_run_parses_json(tester: TesterAgent, mock_client: MagicMock):
    mock_client.complete.return_value = json.dumps(
        {"passed": True, "summary": "all good", "failures": []}
    )
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
        mock_run_tests.return_value = "test output"

        result = await tester.run("goal", [TaskModel(id="1", description="task one")])

    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


async def test_tester_run_fallback_passed(tester: TesterAgent, mock_client: MagicMock):
    mock_client.complete.return_value = "tests passed here"
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
        mock_run_tests.return_value = "test output"

        result = await tester.run("goal", [])

    assert result.passed is True
    assert result.summary == "tests passed here"


async def test_tester_run_fallback_not_passed(tester: TesterAgent, mock_client: MagicMock):
    mock_client.complete.return_value = "something went wrong"
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
        mock_run_tests.return_value = "test output"

        result = await tester.run("goal", [])

    assert result.passed is False
    assert result.summary == "something went wrong"


async def test_run_tests_no_runner(tester: TesterAgent):
    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError,
    ):
        result = await tester._run_tests()

    assert result == "No test runner found."
