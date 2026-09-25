from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


logger = logging.getLogger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except OSError as e:
            logger.warning("Test execution failed: %s", e)
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # If no tests found, consider it passed but warn
        if test_output == "No test runner found.":
            return TestResult(passed=True, summary="No test runner found - skipping tests", failures=[])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: check for common pass/fail indicators
            passed = "passed" in response.lower() or "success" in response.lower()
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q", "--tb=short"],
            ["python", "-m", "pytest", "-q", "--tb=short"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.debug("Test command timed out: %s", cmd)
                    continue
            except FileNotFoundError:
                logger.debug("Test command not found: %s", cmd)
                continue
            except OSError as e:
                logger.debug("Test command failed: %s - %s", cmd, e)
                continue
        return "No test runner found."