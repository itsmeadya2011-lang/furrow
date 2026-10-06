from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or global_settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Parse test output for clear pass/fail signals
        parsed = self._parse_test_output(test_output)
        if parsed is not None:
            return parsed

        # Fall back to LLM interpretation
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _parse_test_output(self, output: str) -> TestResult | None:
        """Parse common test runner output for definitive pass/fail."""
        output_lower = output.lower()

        # pytest patterns
        if "passed" in output_lower and "failed" in output_lower:
            # Look for pytest summary line like "5 passed, 2 failed in 1.23s"
            match = re.search(r"(\d+)\s+passed.*?(\d+)\s+failed", output_lower)
            if match:
                passed, failed = int(match.group(1)), int(match.group(2))
                if failed == 0:
                    return TestResult(passed=True, summary=f"{passed} tests passed", failures=[])
                else:
                    # Extract failure details
                    failures = self._extract_pytest_failures(output)
                    return TestResult(passed=False, summary=f"{passed} passed, {failed} failed", failures=failures)

        # Simple pass/fail patterns
        if "all tests passed" in output_lower or "ok" == output_lower.strip():
            return TestResult(passed=True, summary="All tests passed", failures=[])

        if "error" in output_lower or "fail" in output_lower:
            failures = self._extract_generic_failures(output)
            return TestResult(passed=False, summary="Tests failed", failures=failures)

        return None

    def _extract_pytest_failures(self, output: str) -> list[str]:
        """Extract individual pytest failure messages."""
        failures = []
        # Look for FAILED test lines
        for line in output.split("\n"):
            if "FAILED" in line and "::" in line or "AssertionError" in line or "Error:" in line:
                failures.append(line.strip())
        return failures[:10]  # Limit to 10 failures

    def _extract_generic_failures(self, output: str) -> list[str]:
        """Extract generic failure messages."""
        failures = []
        for line in output.split("\n"):
            line_lower = line.lower()
            if any(kw in line_lower for kw in ["error", "fail", "exception", "traceback"]):
                failures.append(line.strip())
        return failures[:10]

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
                    combined = stdout.decode() + stderr.decode()
                    if combined.strip():
                        return combined
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:
                # Log unexpected errors but continue trying other test runners
                import logging
                logging.getLogger(__name__).debug("Test runner %s failed: %s", cmd[0], e)
                continue
        return "No test runner found."
