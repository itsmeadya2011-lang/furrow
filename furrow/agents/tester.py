from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.max_fix_attempts = 3

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # If tests passed, return success
        if self._tests_passed(test_output):
            return TestResult(passed=True, summary="All tests passed", failures=[])

        # Try to fix failures
        for attempt in range(self.max_fix_attempts):
            logger.info("tester_fix_attempt", attempt=attempt + 1, max_attempts=self.max_fix_attempts)
            
            prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
            
            try:
                data = json.loads(response)
                test_result = TestResult(**data)
            except (json.JSONDecodeError, ValueError):
                test_result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])
            
            if test_result.passed:
                return test_result
            
            # Try to fix the failures by creating fix tasks
            if test_result.failures:
                fix_result = await self._attempt_fix(test_result.failures, goal)
                if fix_result:
                    # Re-run tests after fix
                    test_output = await self._run_tests()
                    if self._tests_passed(test_output):
                        return TestResult(passed=True, summary="Tests passed after fix", failures=[])
        
        return TestResult(passed=False, summary="Failed to fix tests after max attempts", failures=test_result.failures)

    def _tests_passed(self, output: str) -> bool:
        """Quick check if tests appear to pass based on output."""
        output_lower = output.lower()
        # Common pass indicators
        if "passed" in output_lower and "failed" not in output_lower:
            return True
        if "ok" in output_lower and "error" not in output_lower and "fail" not in output_lower:
            return True
        return False

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
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
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    logger.info("tests_run", cmd=cmd[0], exit_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("tests_timeout", cmd=cmd[0])
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_runner_not_found", cmd=cmd[0], error=str(e))
                continue
        return "No test runner found."

    async def _attempt_fix(self, failures: list[str], goal: str) -> bool:
        """Attempt to fix test failures by spawning worker tasks."""
        # For now, just return False - the orchestrator will handle re-planning
        # In a more advanced version, this could create specific fix tasks
        return False
