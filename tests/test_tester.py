import asyncio
from unittest.mock import AsyncMock

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient


class FakeLLM(LLMClient):
    async def complete(self, prompt, system="", model=None):
        return '{"passed": true, "summary": "ok", "failures": []}'


def test_tester_run_returns_passed():
    tester = TesterAgent(client=FakeLLM())
    tester._run_tests = AsyncMock(return_value="ok")
    result = asyncio.run(tester.run("goal", []))
    assert result.passed is True
