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
    def __init__(self, client: LLMClient | None = None, settings_obj: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings_obj or settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except OSError as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Parse test output for structured failures
        parsed_failures = self._parse_test_failures(test_output)

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            # Merge parsed failures if LLM didn't catch them
            if not result.passed and not result.failures and parsed_failures:
                result.failures = parsed_failures
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(
                passed=passed,
                summary=response,
                failures=parsed_failures if not passed else [],
            )

    async def _run_tests(self) -> str:
        candidates = [
            (["pytest", "-q", "--tb=short"], "pytest"),
            (["python", "-m", "pytest", "-q", "--tb=short"], "pytest"),
            (["npm", "test", "--", "--silent"], "npm"),
            (["pnpm", "test", "--", "--silent"], "pnpm"),
            (["yarn", "test", "--silent"], "yarn"),
            (["cargo", "test", "-q"], "cargo"),
            (["go", "test", "./..."], "go"),
        ]
        for cmd, runner in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    return f"[{runner}] exit_code={proc.returncode}\n{output}"
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, OSError):
                continue
        return "No test runner found."

    def _parse_test_failures(self, output: str) -> list[str]:
        """Parse common test output formats for failures."""
        failures = []

        # Pytest failures
        pytest_failures = re.findall(r"FAILED\s+(.+?)(?:\s+-\s+|$)", output)
        for f in pytest_failures:
            failures.append(f"pytest: {f.strip()}")

        # Assertion errors
        assertions = re.findall(r"AssertionError: (.+)", output)
        for a in assertions:
            failures.append(f"AssertionError: {a.strip()}")

        # General error patterns
        errors = re.findall(r"Error: (.+)", output)
        for e in errors:
            failures.append(f"Error: {e.strip()}")

        # Failed test counts
        failed_match = re.search(r"(\d+)\s+failed", output)
        if failed_match and not failures:
            failures.append(f"{failed_match.group(1)} tests failed")

        return failures
