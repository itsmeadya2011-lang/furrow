from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("tester.started", tasks=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("tester.run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Parse test output for structured failures
        parsed_failures = self._parse_test_output(test_output)

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        if parsed_failures:
            prompt += f"\nParsed failures:\n{json.dumps(parsed_failures, indent=2)}\n"

        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            # Merge parsed failures if LLM missed any
            if parsed_failures and not result.failures:
                result.failures = parsed_failures
                result.passed = False
            logger.info("tester.completed", passed=result.passed, failures=len(result.failures))
            return result
        except (json.JSONDecodeError, ValueError):
            # Fallback: check if tests passed based on output
            passed = self._infer_passed(test_output)
            logger.warning("tester.parse_failed", passed=passed, failures=len(parsed_failures))
            return TestResult(
                passed=passed,
                summary=test_output[:500] if test_output else "No test output",
                failures=parsed_failures if not passed else []
            )

    def _parse_test_output(self, output: str) -> list[str]:
        """Parse common test framework outputs for failures."""
        failures = []

        if not output:
            return failures

        # pytest patterns
        pytest_failures = re.findall(r'FAILED\s+(.+?)\s+-', output)
        failures.extend([f"pytest: {f}" for f in pytest_failures])

        # pytest short traceback
        pytest_short = re.findall(r'(_+.*?FAILED.*?)(?=_+|$)', output, re.DOTALL)
        for match in pytest_short:
            lines = match.strip().split('\n')
            for line in lines:
                if 'FAILED' in line and '::' in line:
                    failures.append(f"pytest: {line.strip()}")

        # Generic error patterns
        error_patterns = [
            r'(Error|FAIL|Failure):\s*(.+)',
            r'(AssertionError|Exception|Error):\s*(.+)',
        ]
        for pattern in error_patterns:
            matches = re.findall(pattern, output, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    failures.append(f"{match[0]}: {match[1]}")
                else:
                    failures.append(match)

        # npm/jest patterns
        jest_failures = re.findall(r'●\s+(.+?)\n', output)
        failures.extend([f"jest: {f}" for f in jest_failures])

        # cargo test patterns
        cargo_failures = re.findall(r'test\s+(\S+)\s+...\s+FAILED', output)
        failures.extend([f"cargo: {f}" for f in cargo_failures])

        # go test patterns
        go_failures = re.findall(r'--- FAIL: (\S+)', output)
        failures.extend([f"go: {f}" for f in go_failures])

        return list(set(failures))  # Deduplicate

    def _infer_passed(self, output: str) -> bool:
        """Infer if tests passed from output when JSON parsing fails."""
        if not output:
            return False
        
        output_lower = output.lower()
        # Common pass indicators
        pass_indicators = ["all tests passed", "passed", "ok", "success"]
        # Common fail indicators (higher priority)
        fail_indicators = ["failed", "error", "failure", "failing", "segfault", "panic"]
        
        for fail in fail_indicators:
            if fail in output_lower:
                return False
        
        for passed in pass_indicators:
            if passed in output_lower:
                return True
        
        return False

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
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        logger.warning("tester.no_runner_found")
        return "No test runner found."