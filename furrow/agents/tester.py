from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    pass


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, config: Settings | None = None) -> None:
        self.client = client or LLMClient()
        self.config = config or client.settings if client else None

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # First, do a quick parse to determine pass/fail without LLM
        quick_result = self._parse_test_output(test_output)
        if quick_result.passed:
            return quick_result

        # If tests failed, use LLM to analyze failures in detail
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback to quick parse result
            return quick_result

    def _parse_test_output(self, output: str) -> TestResult:
        """Parse test output to determine pass/fail without LLM."""
        if not output or output.strip() == "No test runner found.":
            return TestResult(
                passed=False,
                summary="No tests found or no test runner available",
                failures=["No test runner found"]
            )

        # Check for common pass indicators
        pass_indicators = [
            "passed" in output.lower() and "failed" not in output.lower(),
            "all tests passed" in output.lower(),
            "ok" == output.strip().lower(),
            re.search(r'\d+\s+passed', output.lower()) and not re.search(r'\d+\s+failed', output.lower()),
        ]

        # Check for common fail indicators
        fail_indicators = [
            "failed" in output.lower(),
            "error" in output.lower() and "passed" not in output.lower(),
            "FAILED" in output,
            "ERROR" in output,
        ]

        # Extract failures
        failures = []
        if "failed" in output.lower() or "error" in output.lower():
            # Try to extract specific failure lines
            lines = output.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['failed', 'error', 'fail:', 'error:']):
                    if len(line.strip()) > 5 and len(line.strip()) < 200:
                        failures.append(line.strip())
        
        # If we couldn't extract specific failures, use the whole output
        if not failures and fail_indicators:
            failures = [output[:500]]

        passed = any(pass_indicators) and not any(fail_indicators)
        
        if passed:
            return TestResult(passed=True, summary="All tests passed (parsed from output)", failures=[])
        else:
            return TestResult(
                passed=False,
                summary=f"Tests failed (parsed from output): {output[:200]}",
                failures=failures[:10]  # Limit failures
            )

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
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
