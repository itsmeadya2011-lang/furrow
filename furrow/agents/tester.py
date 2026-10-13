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
        self.settings = settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=f"Test runner error: {e}", failures=[str(e)])

        # Also parse test output directly for common patterns
        direct_result = self._parse_test_output(test_output)
        if direct_result is not None:
            return direct_result

        # Fall back to LLM evaluation
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            # Try to extract JSON from response
            extracted = self._extract_json(response)
            if extracted:
                try:
                    return TestResult(**extracted)
                except Exception:
                    pass
            # Last resort: heuristic based on keywords
            passed = "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(passed=passed, summary=response, failures=[] if passed else [response])

    def _parse_test_output(self, output: str) -> TestResult | None:
        """Parse common test runner output formats directly."""
        if not output or output == "No test runner found.":
            return None
            
        output_lower = output.lower()
        
        # pytest patterns
        if "passed" in output_lower or "failed" in output_lower or "error" in output_lower:
            # Look for pytest summary line like "5 passed, 2 failed in 1.23s"
            passed_match = re.search(r'(\d+)\s+passed', output)
            failed_match = re.search(r'(\d+)\s+failed', output)
            error_match = re.search(r'(\d+)\s+error', output)
            
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            errors = int(error_match.group(1)) if error_match else 0
            
            if passed > 0 or failed > 0 or errors > 0:
                total_failures = failed + errors
                if total_failures == 0:
                    return TestResult(passed=True, summary=output.strip(), failures=[])
                
                # Extract failure details
                failures = self._extract_pytest_failures(output)
                return TestResult(
                    passed=False,
                    summary=f"{passed} passed, {total_failures} failed",
                    failures=failures
                )
        
        # npm/jest patterns
        if "test suite" in output_lower or "tests passed" in output_lower:
            pass_match = re.search(r'(\d+)\s+(?:tests?|passing)', output)
            fail_match = re.search(r'(\d+)\s+(?:failing|failed)', output)
            
            if pass_match or fail_match:
                passed = int(pass_match.group(1)) if pass_match else 0
                failed = int(fail_match.group(1)) if fail_match else 0
                
                if failed == 0:
                    return TestResult(passed=True, summary=output.strip(), failures=[])
                
                failures = self._extract_js_failures(output)
                return TestResult(
                    passed=False,
                    summary=f"{passed} passed, {failed} failed",
                    failures=failures
                )
        
        # Go test patterns
        if "--- pass:" in output_lower or "--- fail:" in output_lower:
            pass_count = len(re.findall(r'--- pass:', output_lower))
            fail_count = len(re.findall(r'--- fail:', output_lower))
            
            if fail_count == 0:
                return TestResult(passed=True, summary=output.strip(), failures=[])
            
            failures = self._extract_go_failures(output)
            return TestResult(
                passed=False,
                summary=f"{pass_count} passed, {fail_count} failed",
                failures=failures
            )
        
        return None

    def _extract_pytest_failures(self, output: str) -> list[str]:
        """Extract failure details from pytest output."""
        failures = []
        # Look for FAILED test lines
        for line in output.split('\n'):
            if 'FAILED' in line or 'ERROR' in line:
                failures.append(line.strip())
        # Also look for tracebacks
        if not failures and 'traceback' in output.lower():
            # Find traceback sections
            parts = re.split(r'\n\s*\n', output)
            for part in parts:
                if 'traceback' in part.lower() or 'assert' in part.lower():
                    failures.append(part.strip()[:500])  # Limit length
        return failures[:10]  # Limit number of failures

    def _extract_js_failures(self, output: str) -> list[str]:
        """Extract failure details from JS test output."""
        failures = []
        for line in output.split('\n'):
            if any(kw in line.lower() for kw in ['fail', 'error', '✕', '✗']):
                failures.append(line.strip())
        return failures[:10]

    def _extract_go_failures(self, output: str) -> list[str]:
        """Extract failure details from Go test output."""
        failures = []
        for line in output.split('\n'):
            if '--- FAIL:' in line or 'FAIL\t' in line:
                failures.append(line.strip())
        return failures[:10]

    def _extract_json(self, text: str) -> dict | None:
        """Try to extract JSON from text."""
        # Try to find JSON object
        brace_count = 0
        start = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start != -1:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        continue
        return None

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
