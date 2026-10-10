from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

configure_logging()
logger = get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("testing_started", task_count=len(tasks))
        
        # Run tests, linting, and type checking
        all_outputs = []
        
        test_output = await self._run_tests()
        all_outputs.append(("Tests", test_output))
        
        lint_output = await self._run_linting()
        if lint_output:
            all_outputs.append(("Linting", lint_output))
        
        type_output = await self._run_type_check()
        if type_output:
            all_outputs.append(("Type Checking", type_output))
        
        combined_output = "\n\n".join(f"=== {name} ===\n{output}" for name, output in all_outputs)
        
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{combined_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        
        # First try to parse JSON response
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("testing_completed", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("testing_parse_failed", error=str(e))
            # Fallback: parse test output directly
            return self._parse_test_output(combined_output)

    def _parse_test_output(self, output: str) -> TestResult:
        """Parse test output directly to determine pass/fail."""
        # Check for common failure patterns
        failure_patterns = [
            r"(?i)failed",
            r"(?i)error",
            r"(?i)traceback",
            r"(?i)exception",
            r"(?i)assertion.*fail",
            r"(?i)test.*fail",
        ]
        
        # Check for success patterns
        success_patterns = [
            r"(?i)passed",
            r"(?i)all.*test.*pass",
            r"(?i)ok$",
            r"(?i)success",
        ]
        
        has_failure = any(re.search(pattern, output) for pattern in failure_patterns)
        has_success = any(re.search(pattern, output) for pattern in success_patterns)
        
        if has_failure and not has_success:
            # Extract failure lines
            failures = []
            for line in output.split("\n"):
                if any(re.search(p, line) for p in failure_patterns):
                    failures.append(line.strip())
            return TestResult(
                passed=False,
                summary=f"Tests failed with {len(failures)} issues",
                failures=failures[:10]  # Limit failures
            )
        elif has_success:
            return TestResult(passed=True, summary="Tests passed", failures=[])
        else:
            # Unclear - assume failed for safety
            return TestResult(passed=False, summary="Could not determine test status", failures=[output[:500]])

    async def _run_tests(self) -> str:
        logger.debug("testing_searching_runner")
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
                    output = stdout.decode() + stderr.decode()
                    logger.debug("testing_runner_found", cmd=cmd[0], output_len=len(output))
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("testing_runner_timeout", cmd=cmd[0])
                    continue
            except (FileNotFoundError, Exception):
                continue
        logger.warning("testing_no_runner_found")
        return "No test runner found."

    async def _run_linting(self) -> str:
        """Run linting tools and return output."""
        logger.debug("testing_running_linting")
        candidates = [
            ["ruff", "check", "."],
            ["flake8", "."],
            ["pylint", "."],
            ["eslint", "."],
            ["golangci-lint", "run"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        logger.debug("linting_runner_found", cmd=cmd[0], output_len=len(output))
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return ""

    async def _run_type_check(self) -> str:
        """Run type checking tools and return output."""
        logger.debug("testing_running_type_check")
        candidates = [
            ["mypy", "."],
            ["pyright", "."],
            ["tsc", "--noEmit"],
            ["go", "vet", "./..."],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        logger.debug("type_check_runner_found", cmd=cmd[0], output_len=len(output))
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return ""
