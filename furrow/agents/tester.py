from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("furrow.tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_run_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # If tests passed, return success
        if self._tests_passed(test_output):
            logger.info("tests_passed")
            return TestResult(passed=True, summary="All tests passed", failures=[])

        # Tests failed - try to fix them
        logger.warning("tests_failed", output=test_output[:500])
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        
        try:
            data = json.loads(response)
            result = TestResult(**data)
            # If LLM says tests passed but they actually didn't, trust the test output
            if result.passed and not self._tests_passed(test_output):
                logger.warning("llm_said_pass_but_tests_fail")
                return TestResult(passed=False, summary="Tests still failing", failures=self._extract_failures(test_output))
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("test_result_parse_failed", error=str(e))
            # Fallback: parse test output directly
            return TestResult(
                passed=self._tests_passed(test_output),
                summary=test_output[:500],
                failures=self._extract_failures(test_output)
            )

    def _tests_passed(self, output: str) -> bool:
        """Heuristic to determine if tests passed from output."""
        output_lower = output.lower()
        # Check for common failure indicators
        if any(word in output_lower for word in ["failed", "error", "failure", "traceback"]):
            # But check if it's just "0 failed" or similar
            if "0 failed" in output_lower or "0 errors" in output_lower:
                return True
            return False
        # Check for success indicators
        if any(phrase in output_lower for phrase in ["passed", "ok", "success", "all tests passed"]):
            return True
        return False

    def _extract_failures(self, output: str) -> list[str]:
        """Extract failure messages from test output."""
        failures = []
        lines = output.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ["failed", "error:", "failure:", "assertionerror", "traceback"]):
                failures.append(line.strip())
        return failures[:10]  # Limit to 10 failures

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
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
