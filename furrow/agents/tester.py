from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, config: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=config or settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except OSError as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # First, try to parse test results directly
        parsed = self._parse_test_output(test_output)
        if parsed is not None:
            return parsed

        # Fallback to LLM interpretation
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

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
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return "No test runner found."

    def _parse_test_output(self, output: str) -> TestResult | None:
        """Parse common test runner outputs to determine pass/fail."""
        if not output or "No test runner found" in output:
            return None

        # pytest patterns
        if "passed" in output and "failed" in output:
            # Look for "X passed, Y failed" pattern
            match = re.search(r"(\d+)\s+passed.*?(\d+)\s+failed", output)
            if match:
                passed = int(match.group(1))
                failed = int(match.group(2))
                if failed == 0:
                    return TestResult(passed=True, summary=f"{passed} tests passed", failures=[])
                else:
                    # Extract failure lines
                    failures = self._extract_pytest_failures(output)
                    return TestResult(
                        passed=False,
                        summary=f"{passed} passed, {failed} failed",
                        failures=failures,
                    )

        # Check for explicit pass/fail indicators
        lower = output.lower()
        if "all tests passed" in lower or "test suite passed" in lower:
            return TestResult(passed=True, summary="All tests passed", failures=[])
        if "test failed" in lower or "tests failed" in lower or "failure" in lower:
            return TestResult(
                passed=False,
                summary="Tests failed",
                failures=self._extract_generic_failures(output),
            )

        return None

    def _extract_pytest_failures(self, output: str) -> list[str]:
        """Extract failure details from pytest output."""
        failures = []
        lines = output.split("\n")
        in_failure = False
        current_failure: list[str] = []

        for line in lines:
            if "FAILED" in line and "::" in line:
                if current_failure:
                    failures.append("\n".join(current_failure))
                current_failure = [line.strip()]
                in_failure = True
            elif in_failure:
                if line.strip() == "" or line.startswith("="):
                    if current_failure:
                        failures.append("\n".join(current_failure))
                        current_failure = []
                    in_failure = False
                else:
                    current_failure.append(line.strip())

        if current_failure:
            failures.append("\n".join(current_failure))

        return failures[:10]  # Limit to 10 failures

    def _extract_generic_failures(self, output: str) -> list[str]:
        """Extract failures from generic test output."""
        failures = []
        lines = output.split("\n")
        for line in lines:
            if any(keyword in line.lower() for keyword in ["error", "fail", "exception", "traceback"]):
                failures.append(line.strip())
        return failures[:10]
