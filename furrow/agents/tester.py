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
        test_command = ""
        try:
            test_output, test_command = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Parse test output for structured failure info
        parsed_failures = self._parse_test_failures(test_output, test_command)
        
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest command: {test_command}\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            # Merge parsed failures with LLM analysis
            if parsed_failures and not result.failures:
                result.failures = parsed_failures
                result.passed = False
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(passed=passed, summary=response, failures=parsed_failures)

    def _parse_test_failures(self, output: str, command: str) -> list[str]:
        failures = []
        
        # Pytest failures
        if "pytest" in command or "python -m pytest" in command:
            # Look for FAILED lines
            failed_matches = re.findall(r"FAILED\s+(.+?)(?:\s+-|\s*$)", output)
            for match in failed_matches:
                failures.append(f"pytest: {match.strip()}")
            
            # Look for assertion errors
            assertion_matches = re.findall(r"AssertionError:?\s*(.+?)(?:\n|$)", output)
            for match in assertion_matches:
                failures.append(f"AssertionError: {match.strip()}")
            
            # Look for error tracebacks
            error_matches = re.findall(r"(Error|Exception):\s*(.+?)(?:\n|$)", output)
            for err_type, msg in error_matches:
                failures.append(f"{err_type}: {msg.strip()}")

        # Generic patterns
        if not failures:
            # Look for any FAILED or ERROR lines
            for line in output.splitlines():
                line = line.strip()
                if line.startswith("FAILED") or line.startswith("ERROR") or "FAILED" in line:
                    failures.append(line)
        
        # If still no failures but tests didn't pass
        if not failures and ("FAILED" in output or "ERROR" in output or "failed" in output.lower()):
            failures.append("Test output indicates failures (see full output)")
        
        return failures[:10]  # Limit to 10 failures

    async def _run_tests(self) -> tuple[str, str]:
        candidates = [
            (["pytest", "-q", "--tb=short"], "pytest -q --tb=short"),
            (["python", "-m", "pytest", "-q", "--tb=short"], "python -m pytest -q --tb=short"),
            (["npm", "test", "--", "--silent"], "npm test --silent"),
            (["pnpm", "test", "--", "--silent"], "pnpm test --silent"),
            (["yarn", "test", "--silent"], "yarn test --silent"),
            (["cargo", "test", "-q"], "cargo test -q"),
            (["go", "test", "./..."], "go test ./..."),
        ]
        for cmd, cmd_str in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode(), cmd_str
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found.", "none"
