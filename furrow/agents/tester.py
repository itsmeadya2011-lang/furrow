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
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # First, try to parse test output directly for common patterns
        parsed = self._parse_test_output(test_output)
        if parsed is not None:
            return parsed

        # Fall back to LLM analysis
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Last resort: simple heuristic
            passed = "passed" in response.lower() or "all tests passed" in test_output.lower()
            failures = self._extract_failures(test_output) if not passed else []
            return TestResult(
                passed=passed,
                summary=response[:200] if not passed else "Tests passed",
                failures=failures
            )

    def _parse_test_output(self, output: str) -> TestResult | None:
        """Try to parse common test runner outputs directly."""
        output_lower = output.lower()
        
        # pytest patterns
        if "pytest" in output_lower or "passed" in output_lower or "failed" in output_lower:
            # Look for "X passed, Y failed" pattern
            match = re.search(r'(\d+)\s+passed', output_lower)
            passed_count = int(match.group(1)) if match else 0
            match = re.search(r'(\d+)\s+failed', output_lower)
            failed_count = int(match.group(1)) if match else 0
            
            if passed_count > 0 or failed_count > 0:
                failures = self._extract_failures(output)
                return TestResult(
                    passed=failed_count == 0,
                    summary=f"{passed_count} passed, {failed_count} failed",
                    failures=failures
                )
        
        # Generic patterns
        if "all tests passed" in output_lower or "ok" == output_lower.strip():
            return TestResult(passed=True, summary="All tests passed", failures=[])
        
        if "error" in output_lower or "failure" in output_lower:
            failures = self._extract_failures(output)
            return TestResult(passed=False, summary="Tests failed", failures=failures)
        
        return None

    def _extract_failures(self, output: str) -> list[str]:
        """Extract failure messages from test output."""
        failures = []
        lines = output.split('\n')
        
        # Common failure patterns
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['failed', 'error', 'assertion', 'traceback']):
                if len(line) > 10 and len(line) < 200:  # Reasonable length
                    failures.append(line)
        
        # If no specific failures found, return truncated output
        if not failures and output.strip():
            failures = [output[:500]]
        
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
