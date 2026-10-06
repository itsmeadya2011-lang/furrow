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

logger = get_logger("tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("testing_start", goal=goal[:100], num_tasks=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_execution_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        logger.debug("test_output", output=test_output[:500])
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        logger.debug("tester_response", response=response[:500])
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("test_result", passed=result.passed, summary=result.summary, num_failures=len(result.failures))
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            logger.warning("test_parse_failed", response=response[:200], inferred_passed=passed)
            return TestResult(passed=passed, summary=response, failures=[])

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
                logger.debug("trying_test_runner", command=cmd)
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    logger.info("test_runner_found", command=cmd, exit_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_runner_timeout", command=cmd)
                    continue
            except FileNotFoundError:
                logger.debug("test_runner_not_found", command=cmd)
                continue
            except Exception as e:
                logger.debug("test_runner_error", command=cmd, error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."