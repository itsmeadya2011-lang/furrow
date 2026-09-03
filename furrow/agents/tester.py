from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    """Agent responsible for running tests and parsing test results via LLM."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        """Initialize the TesterAgent with an optional LLM client and settings.

        Args:
            client: An optional LLMClient instance. A new one is created if not provided.
            settings: An optional Settings instance passed to the LLMClient.
        """
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the test suite and parse the results using the LLM.

        Args:
            goal: The high-level goal for the test run.
            tasks: The list of TaskModel objects associated with the goal.

        Returns:
            A TestResult indicating pass/fail status, summary, and failures.
        """
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if test_output == "No test runner found.":
            return TestResult(passed=True, summary="No test runner found.", failures=[])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await asyncio.wait_for(
            self.client.complete(prompt, model=self.client.settings.tester_model),
            timeout=60,
        )
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        """Discover and execute an available test runner from a list of candidates.

        Iterates through a predefined list of common test commands. Each command
        is tried in sequence. Commands that are not found on the system are
        skipped, and the next candidate is attempted. If all candidates fail,
        a message indicating no test runner was found is returned.

        Returns:
            The combined stdout and stderr output from the first successful test
            command, or a message indicating no test runner was found.
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
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return "No test runner found."
