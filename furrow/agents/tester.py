from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from furrow.agents._util import parse_json_response
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger(__name__)

_TEST_CANDIDATES = [
    ["pytest", "-q"],
    ["python", "-m", "pytest", "-q"],
    ["pnpm", "test", "--", "--silent"],
    ["yarn", "test", "--silent"],
    ["npm", "test", "--", "--silent"],
    ["cargo", "test", "-q"],
    ["go", "test", "./..."],
]


class TesterAgent:
    __test__ = False  # not a pytest test class

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - surface any harness error as a failure
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        return self._parse_result(response)

    def _parse_result(self, response: str) -> TestResult:
        data = parse_json_response(response)
        if data:
            fields = {k: v for k, v in data.items() if k in TestResult.model_fields}
            try:
                return TestResult(**fields)
            except (ValueError, TypeError):
                pass
        passed = "passed" in response.lower()
        return TestResult(passed=passed, summary=response.strip(), failures=[])

    async def _run_tests(self) -> str:
        workspace = self.client.settings.workspace
        for cmd in _TEST_CANDIDATES:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace),
                )
            except (FileNotFoundError, OSError):
                continue
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.client.settings.test_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                continue
            return stdout.decode(errors="replace") + stderr.decode(errors="replace")
        return "No test runner found."
