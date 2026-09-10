from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING

import structlog
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_execution_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Parse test output for common patterns
        parsed = self._parse_test_output(test_output)
        
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            # Merge parsed failures if LLM missed them
            if parsed["failures"] and not result.failures:
                result.failures = parsed["failures"]
                result.passed = False
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("test_result_parse_failed", error=str(e), response=response[:200])
            # Fallback to parsed result
            return TestResult(
                passed=parsed["passed"],
                summary=parsed["summary"],
                failures=parsed["failures"]
            )

    def _parse_test_output(self, output: str) -> dict[str, Any]:
        """Parse common test output formats to extract pass/fail info."""
        if not output or output == "No test runner found.":
            return {"passed": False, "summary": "No test runner found", "failures": ["No test runner found"]}

        # Check for pytest output patterns
        passed_pattern = re.search(r'(\d+)\s+passed', output)
        failed_pattern = re.search(r'(\d+)\s+failed', output)
        error_pattern = re.search(r'(\d+)\s+error', output)
        
        # Check for other test frameworks
        if "OK" in output and "FAILED" not in output:
            return {"passed": True, "summary": "Tests passed", "failures": []}
        
        if "FAILED" in output or "ERROR" in output:
            # Extract failure lines
            failures = []
            for line in output.split('\n'):
                if 'FAILED' in line or 'ERROR' in line or 'AssertionError' in line:
                    failures.append(line.strip())
            return {
                "passed": False,
                "summary": f"Tests failed: {len(failures)} failures",
                "failures": failures[:10]  # Limit to first 10
            }

        if passed_pattern and not failed_pattern and not error_pattern:
            return {"passed": True, "summary": f"{passed_pattern.group(1)} tests passed", "failures": []}

        # Default: let LLM decide
        return {"passed": "passed" in output.lower(), "summary": output[:500], "failures": []}

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
