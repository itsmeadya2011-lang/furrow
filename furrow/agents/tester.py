from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Parse test output for common patterns
        parsed_failures = self._parse_test_output(test_output)
        
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            # Merge parsed failures with LLM-detected failures
            if parsed_failures and not result.failures:
                result.failures = parsed_failures
                result.passed = False
            return result
        except (json.JSONDecodeError, ValueError):
            # Fallback: use parsed failures
            if parsed_failures:
                return TestResult(passed=False, summary="Tests failed (parsed from output)", failures=parsed_failures)
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _parse_test_output(self, output: str) -> list[str]:
        """Parse common test output formats for failures."""
        failures = []
        
        # Pytest failures
        pytest_failures = re.findall(r"(FAILED|ERROR)\s+(.+?)(?:\s+[-=])", output)
        for status, test_name in pytest_failures:
            failures.append(f"{status}: {test_name}")
        
        # Pytest traceback failures
        traceback_failures = re.findall(r"(.+?\.py:\d+):\s*(.+?)(?:\n|$)", output)
        for location, error in traceback_failures:
            if "error" in error.lower() or "fail" in error.lower() or "assert" in error.lower():
                failures.append(f"{location}: {error}")
        
        # Generic assertions
        assertion_failures = re.findall(r"AssertionError: (.+)", output)
        for err in assertion_failures:
            failures.append(f"AssertionError: {err}")
        
        # Python tracebacks
        if "Traceback (most recent call last):" in output:
            # Extract the last error line
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "Error:" in line or "Exception:" in line:
                    failures.append(line.strip())
                    break
        
        # JavaScript/Node test failures
        js_failures = re.findall(r"✕\s+(.+?)(?:\n|$)", output)
        for err in js_failures:
            failures.append(f"Test failed: {err}")
        
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
