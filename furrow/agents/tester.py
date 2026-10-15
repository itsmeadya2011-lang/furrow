from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
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
        logger.info("Test run started", goal=goal, task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - any test-run failure should degrade gracefully
            logger.error("Test run failed during _run_tests", error=str(e))
            result = TestResult(passed=False, summary=str(e), failures=[str(e)])
            logger.info("test_result", passed=result.passed)
            return result

        logger.info("Test output received", test_output=test_output)
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("test_result", passed=result.passed)
            logger.info("Test run completed", test_result=result.model_dump())
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            result = TestResult(passed=passed, summary=response, failures=[])
            logger.info("test_result", passed=result.passed)
            logger.info("Test run completed (fallback)", test_result=result.model_dump())
            return result

    async def _run_tests(self) -> str:
        logger.debug("Running test candidates")
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
                    logger.debug("Test command succeeded", cmd=cmd, output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    logger.warning("Test command timed out", cmd=cmd)
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception) as e:  # noqa: BLE001 - any subprocess failure should degrade gracefully
                logger.debug("Test command failed", cmd=cmd, error=str(e))
                continue
        logger.warning("No test runner found")
        return "No test runner found."
