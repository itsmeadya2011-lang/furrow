from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        test_runner = ""
        try:
            test_output, test_runner = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Parse test output for quick feedback
        parsed = self._parse_test_output(test_output, test_runner)
        if parsed["quick_failures"]:
            return TestResult(passed=False, summary=parsed["summary"], failures=parsed["quick_failures"])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest runner: {test_runner}\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: check if output contains obvious failures
            if "FAILED" in test_output.upper() or "ERROR" in test_output.upper():
                return TestResult(passed=False, summary="Tests appear to have failed", failures=[test_output[:500]])
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _parse_test_output(self, output: str, runner: str) -> dict:
        """Parse test output for quick failure detection without LLM."""
        failures = []
        summary = ""
        
        if "pytest" in runner:
            # Parse pytest output
            failed_matches = re.findall(r'FAILED\s+(.+?)\s+[-=]', output)
            if failed_matches:
                failures = [f"Pytest failure: {m}" for m in failed_matches[:10]]
            
            # Look for error summary
            error_matches = re.findall(r'ERROR\s+(.+?)\s+[-=]', output)
            if error_matches:
                failures.extend([f"Pytest error: {m}" for m in error_matches[:10]])
            
            # Get summary line
            summary_match = re.search(r'(passed|failed|error).*?in\s+\d+\.\d+s', output)
            summary = summary_match.group(0) if summary_match else "Pytest completed"
            
        elif "npm" in runner or "yarn" in runner or "pnpm" in runner:
            # Parse npm/yarn/pnpm test output
            if "failed" in output.lower() or "error" in output.lower():
                failures = ["JavaScript test failures detected"]
            summary = "JavaScript tests completed"
            
        elif "cargo" in runner:
            # Parse cargo test output
            failed_matches = re.findall(r'test\s+(.+?)\s+...\s+FAILED', output)
            if failed_matches:
                failures = [f"Cargo test failure: {m}" for m in failed_matches[:10]]
            summary = "Cargo tests completed"
            
        elif "go test" in runner:
            # Parse go test output
            if "FAIL" in output:
                failures = ["Go test failures detected"]
            summary = "Go tests completed"
        
        quick_failures = failures[:5]  # Limit to first 5 for quick feedback
        
        return {
            "quick_failures": quick_failures,
            "summary": summary
        }

    async def _run_tests(self) -> tuple[str, str]:
        candidates = [
            (["pytest", "-q"], "pytest"),
            (["python", "-m", "pytest", "-q"], "pytest"),
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
                    return stdout.decode() + stderr.decode(), runner
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found.", "none"
