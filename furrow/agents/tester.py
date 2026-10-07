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


logger = get_logger("tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("testing_started", task_count=len(tasks))
        test_output = ""
        test_runner = "none"
        try:
            test_output, test_runner = await self._run_tests()
            logger.info("tests_executed", runner=test_runner, output_length=len(test_output))
        except Exception as e:
            logger.error("test_execution_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Quick heuristic check before LLM analysis
        quick_result = self._quick_check(test_output, test_runner)
        if quick_result is not None:
            logger.info("quick_check_result", passed=quick_result.passed, summary=quick_result.summary)
            return quick_result

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("test_analysis_complete", passed=result.passed, failure_count=len(result.failures))
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            logger.warning("test_analysis_parse_failed", response=response[:200])
            return TestResult(passed=passed, summary=response, failures=[])

    def _quick_check(self, output: str, runner: str) -> TestResult | None:
        """Quick heuristic to determine test status without LLM."""
        output_lower = output.lower()
        
        if runner == "pytest":
            # Check for pytest patterns
            if "passed" in output_lower and "failed" not in output_lower and "error" not in output_lower:
                # Extract summary like "5 passed in 0.45s"
                match = re.search(r"(\d+)\s+passed", output_lower)
                if match:
                    return TestResult(passed=True, summary=f"{match.group(1)} tests passed", failures=[])
            if "failed" in output_lower or "error" in output_lower:
                failures = self._extract_pytest_failures(output)
                return TestResult(passed=False, summary="pytest failures detected", failures=failures)
        
        if runner in ("npm", "pnpm", "yarn"):
            if "passed" in output_lower and "failed" not in output_lower:
                return TestResult(passed=True, summary="npm tests passed", failures=[])
            if "failed" in output_lower:
                return TestResult(passed=False, summary="npm test failures", failures=[output[:500]])
        
        if runner == "cargo":
            if "test result: ok" in output_lower:
                return TestResult(passed=True, summary="cargo tests passed", failures=[])
            if "test result: failed" in output_lower:
                return TestResult(passed=False, summary="cargo test failures", failures=[output[:500]])
        
        if runner == "go":
            if "ok" in output_lower and "fail" not in output_lower:
                return TestResult(passed=True, summary="go tests passed", failures=[])
            if "fail" in output_lower:
                return TestResult(passed=False, summary="go test failures", failures=[output[:500]])
        
        return None

    def _extract_pytest_failures(self, output: str) -> list[str]:
        """Extract failure details from pytest output."""
        failures = []
        # Look for FAILED lines
        for line in output.split("\n"):
            if "FAILED" in line or "ERROR" in line:
                failures.append(line.strip())
        # If no specific failures found, return truncated output
        if not failures:
            failures = [output[-1000:]] if len(output) > 1000 else [output]
        return failures[:10]  # Limit to 10 failures

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
                    logger.warning("test_timeout", runner=runner)
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_runner_not_found", runner=runner, error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found.", "none"
