from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

import structlog
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # If tests passed, return success
        if self._tests_passed(test_output):
            return TestResult(passed=True, summary="All tests passed", failures=[])

        # Tests failed - analyze failures and suggest fixes
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to parse from text
            return TestResult(
                passed="passed" in response.lower() and "failed" not in response.lower(),
                summary=response[:500],
                failures=self._extract_failures(test_output)
            )

    def _tests_passed(self, output: str) -> bool:
        """Quick check if tests appear to have passed."""
        output_lower = output.lower()
        failure_indicators = ["failed", "error", "traceback", "assertionerror", "failures="]
        success_indicators = ["passed", "ok", "success", "all tests passed"]
        
        has_failure = any(ind in output_lower for ind in failure_indicators)
        has_success = any(ind in output_lower for ind in success_indicators)
        
        return has_success and not has_failure

    def _extract_failures(self, output: str) -> list[str]:
        """Extract failure messages from test output."""
        failures = []
        lines = output.split("\n")
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ["failed", "error", "traceback", "assertionerror"]):
                # Include some context
                context = "\n".join(lines[max(0, i-2):i+3])
                failures.append(context[:500])
        return failures[:10]  # Limit to 10 failures

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
                    logger.info("tests_run", cmd=cmd[0], exit_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", cmd=cmd[0])
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_runner_not_found", cmd=cmd[0], error=str(e))
                continue
        return "No test runner found."
