from unittest.mock import AsyncMock

from furrow.config import Settings, TaskModel
from furrow.agents.tester import TesterAgent
from furrow.llm import LLMClient


def _make_tester(complete_return):
    client = LLMClient(settings=Settings())
    client.complete = AsyncMock(return_value=complete_return)
    return TesterAgent(client=client)


def _subprocess_mock(output=b"tests passed\n"):
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(output, b""))
    return AsyncMock(return_value=proc)


async def test_tester_parse_pass_case():
    with patch("furrow.agents.tester.asyncio.create_subprocess_exec", _subprocess_mock()):
        tester = _make_tester('{"passed": true, "summary": "all good", "failures": []}')
        result = await tester.run("goal", [TaskModel(id="1")])
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


async def test_tester_parse_fail_case():
    with patch("furrow.agents.tester.asyncio.create_subprocess_exec", _subprocess_mock()):
        tester = _make_tester(
            '{"passed": false, "summary": "some failed", "failures": ["test_a", "test_b"]}'
        )
        result = await tester.run("goal", [TaskModel(id="1")])
    assert result.passed is False
    assert result.summary == "some failed"
    assert result.failures == ["test_a", "test_b"]


async def test_tester_no_test_runner_found_is_handled():
    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError("no runner")),
    ):
        tester = _make_tester('{"passed": true, "summary": "no tests", "failures": []}')
        result = await tester.run("goal", [TaskModel(id="1")])
    assert result.passed is True


async def test_tester_exception_is_handled():
    with patch("furrow.agents.tester.asyncio.create_subprocess_exec", _subprocess_mock()):
        tester = _make_tester(None)
        tester.client.complete = AsyncMock(side_effect=RuntimeError("llm down"))
        result = await tester.run("goal", [TaskModel(id="1")])
    assert result.passed is False
    assert "llm down" in result.summary
    assert "llm down" in result.failures
