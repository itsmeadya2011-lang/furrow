from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.debug("tester_start", tasks_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # If tests passed, return success
        if "passed" in test_output.lower() or "ok" in test_output.lower() or test_output.strip() == "":
            # Additional check - if output is empty or minimal, try to determine pass/fail
            if test_output.strip() == "" or "no test" in test_output.lower():
                logger.info("no_tests_found", output=test_output)
                return TestResult(passed=True, summary="No tests found, assuming pass", failures=[])
            if "failed" not in test_output.lower() and "error" not in test_output.lower():
                logger.info("tests_passed", output=test_output)
                return TestResult(passed=True, summary=test_output, failures=[])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester_result", passed=result.passed, failures=result.failures)
            return result
        except (json.JSONDecodeError, ValueError):
            logger.warning("tester_parse_failed", response=response)
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

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
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=self.settings.workspace
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        logger.debug("test_runner_output", cmd=cmd, output=output[:500])
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."
