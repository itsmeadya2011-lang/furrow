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
    """Agent responsible for executing tests and evaluating test results."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        """Initialize the tester agent with an optional LLM client and settings."""
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Execute tests and return a structured test result.

        Runs the project's test suite, feeds the output to an LLM for
        evaluation, and returns a parsed TestResult. If no test runner is
        found, the result is marked as passed with a summary indicating
        there were no tests to run.
        """
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if test_output == "No test runner found. Assuming no tests to run.":
            return TestResult(passed=True, summary="No test runner found. No tests to verify.", failures=[])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            passed_match = re.search(r'\bpassed\b', response.lower())
            failed_match = re.search(r'\bfailed\b', response.lower())
            if passed_match and not failed_match:
                return TestResult(passed=True, summary=response, failures=[])
            elif failed_match and not passed_match:
                return TestResult(passed=False, summary=response, failures=["Failed keyword found"])
            elif failed_match:
                return TestResult(passed=False, summary=response, failures=["Mixed passed/failed keywords found"])
            else:
                return TestResult(
                    passed=False,
                    summary=f"Could not parse test result. Raw response: {response}",
                    failures=["No passed or failed keyword found in response"]
                )

    async def _run_tests(self) -> str:
        """Discover and run the project's test suite using common runners.

        Iterates through a list of common test runner commands (pytest,
        npm, pnpm, yarn, cargo, go). Returns the combined stdout/stderr
        output of the first successful command, a timeout message if a
        runner exceeds the 120s limit, or a special string if no runner
        is found.
        """
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
                    return "Test runner timed out after 120s."
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found. Assuming no tests to run."
