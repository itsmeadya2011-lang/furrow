from __future__ import annotations

from unittest.mock import AsyncMock, patch

from furrow.agents.tester import TesterAgent
from furrow.config import Settings


async def test_run_parses_json_result():
    client = AsyncMock()
    client.settings = Settings(_env_file=None)
    client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "ok", "failures": []}'
    )
    agent = TesterAgent(client=client)
    with patch.object(agent, "_run_tests", return_value="some output"):
        result = await agent.run("test", [])
    assert result.passed is True
    assert result.summary == "ok"


async def test_run_fallback_on_non_json():
    client = AsyncMock()
    client.settings = Settings(_env_file=None)
    client.complete = AsyncMock(return_value="all tests passed")
    agent = TesterAgent(client=client)
    with patch.object(agent, "_run_tests", return_value="some output"):
        result = await agent.run("test", [])
    assert result.passed is True
    assert result.summary == "all tests passed"


async def test_run_test_exception_returns_failure():
    client = AsyncMock()
    client.settings = Settings(_env_file=None)
    agent = TesterAgent(client=client)
    with patch.object(
        agent, "_run_tests", side_effect=RuntimeError("boom")
    ):
        result = await agent.run("test", [])
    assert result.passed is False
    assert "boom" in result.summary


async def test_run_tests_exit_code_surfaced():
    fake_proc = AsyncMock()
    fake_proc.returncode = 1
    fake_proc.communicate.return_value = (b"fail", b"")
    fake_proc.kill = AsyncMock()
    mock_create = AsyncMock(return_value=fake_proc)
    with patch("asyncio.create_subprocess_exec", mock_create):
        agent = TesterAgent()
        result = await agent._run_tests()
    assert "[exit code: 1]" in result
