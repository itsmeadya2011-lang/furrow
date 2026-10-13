from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Parse test results directly instead of using LLM
        return self._parse_test_output(test_output)

    def _parse_test_output(self, output: str) -> TestResult:
        """Parse test output to determine pass/fail."""
        if not output or output == "No test runner found.":
            return TestResult(
                passed=False,
                summary="No test runner found or no tests executed",
                failures=["No test runner found (tried pytest, npm, cargo, go)"]
            )

        # Check for common test failure patterns
        failures = []
        
        # pytest patterns
        if "FAILED" in output or "failed" in output.lower():
            # Extract failed test names
            failed_matches = re.findall(r'(FAILED|ERROR)\s+([\w/\.::]+)', output)
            for match in failed_matches:
                failures.append(f"{match[0]}: {match[1]}")
            
            # Also look for assertion errors
            assertion_matches = re.findall(r'AssertionError: (.+)', output)
            failures.extend([f"AssertionError: {m}" for m in assertion_matches])
        
        # npm test patterns
        if "npm ERR!" in output or "Test failed" in output:
            failures.append("npm tests failed")
        
        # Cargo test patterns
        if "test result: FAILED" in output:
            failures.append("cargo tests failed")
        
        # Go test patterns
        if "FAIL" in output and "PASS" not in output.split("FAIL")[0]:
            failures.append("go tests failed")

        passed = len(failures) == 0
        
        if passed:
            summary = "All tests passed"
        else:
            summary = f"{len(failures)} test failure(s)"
        
        return TestResult(passed=passed, summary=summary, failures=failures)

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
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
