from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("furrow.tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("testing_start", goal_length=len(goal), task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_execution_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        logger.debug("tester_response", response_length=len(response))
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("testing_complete", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            result = TestResult(passed=passed, summary=response, failures=[])
            logger.warning("tester_parse_fallback", passed=passed)
            return result

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
                logger.debug("test_runner_attempt", cmd=cmd)
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    logger.debug("test_runner_complete", cmd=cmd, returncode=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_runner_timeout", cmd=cmd)
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_runner_not_found", cmd=cmd, error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."
