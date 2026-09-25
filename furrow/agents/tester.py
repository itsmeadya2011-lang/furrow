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

        # First, try to parse test results directly
        parsed = self._parse_test_output(test_output)
        if parsed is not None:
            return parsed

        # Fallback to LLM analysis
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _parse_test_output(self, output: str) -> TestResult | None:
        """Parse common test output formats directly."""
        if not output or "No test runner found" in output:
            return None

        # pytest
        if "pytest" in output or "passed" in output or "failed" in output:
            # Look for pytest summary line
            passed_match = re.search(r"(\d+)\s+passed", output)
            failed_match = re.search(r"(\d+)\s+failed", output)
            error_match = re.search(r"(\d+)\s+error", output)

            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            errors = int(error_match.group(1)) if error_match else 0

            if passed > 0 or failed > 0 or errors > 0:
                failures = []
                if failed > 0 or errors > 0:
                    # Extract failure details
                    failure_lines = re.findall(r"(FAILED|ERROR)\s+(.+)", output)
                    failures = [f"{t}: {n}" for t, n in failure_lines[:10]]
                return TestResult(
                    passed=(failed == 0 and errors == 0),
                    summary=f"{passed} passed, {failed} failed, {errors} errors",
                    failures=failures,
                )

        # npm/jest/vitest
        if "npm" in output or "jest" in output or "vitest" in output:
            pass_match = re.search(r"(\d+)\s+pass(?:ing)?", output)
            fail_match = re.search(r"(\d+)\s+fail(?:ing)?", output)
            passed = int(pass_match.group(1)) if pass_match else 0
            failed = int(fail_match.group(1)) if fail_match else 0
            if passed > 0 or failed > 0:
                return TestResult(
                    passed=failed == 0,
                    summary=f"{passed} passing, {failed} failing",
                    failures=[],
                )

        # cargo test
        if "cargo test" in output or "test result:" in output:
            if "test result: ok" in output.lower():
                return TestResult(passed=True, summary="All tests passed", failures=[])
            elif "test result: FAILED" in output:
                return TestResult(passed=False, summary="Tests failed", failures=["Cargo test failures"])

        # go test
        if "go test" in output:
            if "PASS" in output and "FAIL" not in output:
                return TestResult(passed=True, summary="All tests passed", failures=[])
            elif "FAIL" in output:
                return TestResult(passed=False, summary="Tests failed", failures=["Go test failures"])

        return None

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
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
