from __future__ import annotations

import asyncio
import json
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
        except (OSError, asyncio.TimeoutError) as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Parse test output for pass/fail
        passed, failures = self._parse_test_output(test_output)
        
        if passed:
            return TestResult(passed=True, summary="All tests passed", failures=[])
        
        # If tests failed, try to get fix suggestions from LLM
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed=False, summary=test_output, failures=failures)

    def _parse_test_output(self, output: str) -> tuple[bool, list[str]]:
        """Parse test output to determine pass/fail and extract failures."""
        if not output or output == "No test runner found.":
            return False, ["No test runner found or no tests executed"]
        
        # Check for pytest patterns
        if "passed" in output.lower() and "failed" not in output.lower() and re.search(r'\d+\s+passed', output):
            return True, []
        
        # Check for explicit failures
        failures = []
        if "FAILED" in output or "failed" in output.lower():
            # Extract failure lines
            for line in output.split('\n'):
                if 'FAILED' in line or ('failed' in line.lower() and 'test' in line.lower()):
                    failures.append(line.strip())
        
        # If we found explicit failures, return them
        if failures:
            return False, failures[:10]  # Limit to 10 failures
        
        # Check for error patterns
        if "error" in output.lower() or "traceback" in output.lower():
            return False, [output[:500]]
        
        # Default: assume passed if no clear failure
        return True, []

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
            except (FileNotFoundError, OSError, PermissionError):
                continue
        return "No test runner found."
