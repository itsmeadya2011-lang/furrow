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

        # Parse test output for structured info
        parsed = self._parse_test_output(test_output)

        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n"
            f"Test output:\n{test_output}\n\n"
            f"Parsed summary: {parsed}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to extract pass/fail from response
            passed = "passed" in response.lower() and "failed" not in response.lower()
            failures = re.findall(r"FAILED|Error|error|failed", test_output, re.IGNORECASE)
            return TestResult(
                passed=passed,
                summary=response[:500],
                failures=test_output.split("\n")[-10:] if not passed else [],
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
                    if proc.returncode == 0 or output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."

    def _parse_test_output(self, output: str) -> dict:
        """Parse test output for summary statistics."""
        result = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        # pytest patterns
        for line in output.split("\n"):
            if "passed" in line and "failed" in line:
                # e.g., "5 passed, 2 failed in 1.23s"
                import re

                m = re.search(r"(\d+)\s+passed", line)
                if m:
                    result["passed"] = int(m.group(1))
                m = re.search(r"(\d+)\s+failed", line)
                if m:
                    result["failed"] = int(m.group(1))
                m = re.search(r"(\d+)\s+error", line)
                if m:
                    result["errors"] = int(m.group(1))
                m = re.search(r"(\d+)\s+skipped", line)
                if m:
                    result["skipped"] = int(m.group(1))
        return result
