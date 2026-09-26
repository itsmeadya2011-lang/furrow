from __future__ import annotations

from unittest.mock import AsyncMock, patch

from furrow.agents.tester import TesterAgent


async def test_json_parse_fallback(mock_client) -> None:
    mock_client.complete.return_value = "ALL TESTS PASSED"
    tester = TesterAgent(client=mock_client)

    with patch.object(
        tester, "_run_tests", new_callable=AsyncMock, return_value="fake test output"
    ):
        result = await tester.run("build a thing", [])

    assert result.passed is True
    assert result.summary == "ALL TESTS PASSED"
    assert result.failures == []


async def test_test_run_failure(mock_client) -> None:
    tester = TesterAgent(client=mock_client)

    with patch.object(
        tester, "_run_tests", new_callable=AsyncMock, side_effect=RuntimeError("kaboom")
    ):
        result = await tester.run("goal", [])

    assert result.passed is False
    assert "kaboom" in result.summary
    assert "kaboom" in result.failures[0]
    assert len(result.failures) == 1


async def test_no_test_runner_found(mock_client) -> None:
    tester = TesterAgent(client=mock_client)

    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        result = await tester._run_tests()

    assert result == "No test runner found."
