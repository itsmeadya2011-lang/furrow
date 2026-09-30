from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class TesterAgent:
    __test__ = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - treat test runner errors as a failed result
            logger.error("tester.run_tests.fail", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester.interpret.ok", passed=result.passed)
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            logger.warning("tester.interpret.fallback", passed=passed)
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = [
            ("pytest", ["pytest", "-q"]),
            ("python", ["python", "-m", "pytest", "-q"]),
            ("npm", ["npm", "test", "--", "--silent"]),
            ("pnpm", ["pnpm", "test", "--", "--silent"]),
            ("yarn", ["yarn", "test", "--silent"]),
            ("cargo", ["cargo", "test", "-q"]),
            ("go", ["go", "test", "./..."]),
        ]
        for name, cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    logger.info("tester.run_tests.ok", runner=name)
                    return stdout.decode(errors="replace") + stderr.decode(errors="replace")
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("tester.run_tests.timeout", runner=name)
                    continue
            except (FileNotFoundError, OSError):
                continue
        return "No test runner found."
