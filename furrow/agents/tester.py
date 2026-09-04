from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


TEST_RUNNER_TIMEOUT_SECONDS = 120


class TesterAgent:
    """Agent that runs project tests and summarizes the result using an LLM."""

    DEFAULT_TIMEOUT_SECONDS = TEST_RUNNER_TIMEOUT_SECONDS

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else self.DEFAULT_TIMEOUT_SECONDS
        )

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the project's test suite and return a structured TestResult.

        Exceptions raised while running the test command are converted into a
        failing TestResult so that the caller always receives a usable object.
        """
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        """Execute known test runners and return the combined stdout/stderr.

        Returns ``"No test runner found."`` when none of the candidate
        executables are available on the system.
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
            except (FileNotFoundError, PermissionError, NotADirectoryError, IsADirectoryError):
                continue
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds
                )
                return stdout.decode() + stderr.decode()
            except asyncio.TimeoutError:
                proc.kill()
                continue
        return "No test runner found."
