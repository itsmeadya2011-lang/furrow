from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("tester.started", task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
            logger.info("tester.test_run_completed", output_length=len(test_output))
        except Exception as e:
            logger.error("tester.test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        logger.debug("llm.call.started", model=self.client.settings.tester_model)
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        logger.debug("llm.call.completed", response_length=len(response))
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester.evaluation.completed", passed=result.passed, summary=result.summary, failure_count=len(result.failures))
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            logger.warning("tester.evaluation.invalid_json", fallback_passed=passed)
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
        timeout = self.client.settings.test_timeout
        for cmd in candidates:
            if shutil.which(cmd[0]) is None:
                logger.debug("test_runner.not_found", command=cmd[0])
                continue
            logger.info("test_runner.found", command=cmd[0])
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    exit_code = proc.returncode
                    output = stdout.decode() + stderr.decode()
                    logger.info("test_runner.completed", command=cmd[0], exit_code=exit_code, output_length=len(output))
                    return f"Exit code: {exit_code}\n{output}"
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_runner.timeout", command=cmd[0], timeout=timeout)
                    return f"Test runner '{cmd[0]}' timed out after {timeout} seconds."
            except Exception as e:
                logger.debug("test_runner.error", command=cmd[0], error=str(e))
                continue
        logger.warning("test_runner.none_found")
        return "No test runner found."
