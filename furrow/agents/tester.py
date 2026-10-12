from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT, TESTER_FIX_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("Running tests", task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("Test execution failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        logger.debug("Test output received", length=len(test_output))
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("Test result", passed=result.passed, failures_count=len(result.failures))
            return result
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse test result, using fallback", response=response[:200])
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def fix_failures(self, goal: str, failures: list[str], context: str = "") -> str:
        """Generate fix suggestions for test failures."""
        logger.info("Generating fixes for failures", failure_count=len(failures))
        prompt = f"{TESTER_FIX_PROMPT}\n\nGoal: {goal}\n\nFailures:\n" + "\n".join(failures)
        if context:
            prompt += f"\n\nContext:\n{context}"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        return response

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
            try:
                logger.debug("Trying test command", cmd=cmd)
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    logger.debug("Test command completed", cmd=cmd, returncode=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("Test command timed out", cmd=cmd)
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("Test command failed", cmd=cmd, error=str(e))
                continue
        logger.warning("No test runner found")
        return "No test runner found."
