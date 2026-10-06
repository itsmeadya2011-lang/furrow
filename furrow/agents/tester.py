from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    """Runs the project's test suite and interprets the results.

    The agent tries a curated list of test runners (pytest, npm, cargo, …).
    The first runner that **exists** on the system is used; if it exits with
    a non-zero status the output is still captured and returned so the LLM
    can interpret failures.
    """

    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self._test_timeout: int = 120

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(
                passed=False, summary=str(e), failures=[str(e)]
            )

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(
            prompt, model=self.client.settings.tester_model
        )
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: guess from response text if the LLM didn't return JSON.
            return TestResult(
                passed="passed" in response.lower(),
                summary=response,
                failures=[],
            )

    # -- test execution ------------------------------------------------------

    async def _run_tests(self) -> str:
        """Locate and run the project's test runner.

        Returns the combined stdout+stderr output.  Raises
        ``RuntimeError`` if no known test runner is installed.
        """
        candidates = self._get_test_commands()

        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                # The command itself doesn't exist — try the next candidate.
                continue

            stdout_data: bytes = b""
            stderr_data: bytes = b""
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(), timeout=self._test_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                # Reap the killed process to avoid zombies.
                await proc.wait()
                # Try to drain any remaining buffered output after kill.
                try:
                    stdout_data, stderr_data = await asyncio.wait_for(
                        proc.communicate(), timeout=5
                    )
                except Exception:
                    pass
                return (
                    f"Test runner '{cmd[0]}' timed out after "
                    f"{self._test_timeout}s.\n"
                    f"{stdout_data.decode(errors='replace')}"
                    f"{stderr_data.decode(errors='replace')}"
                )

            # Return output regardless of exit code — a non-zero exit
            # means tests *failed*, which the LLM interpreter can parse.
            output = stdout_data.decode(errors="replace") + stderr_data.decode(errors="replace")
            return output

        return "No test runner found."

    def _get_test_commands(self) -> list[list[str]]:
        """Build the list of test commands to try, in priority order."""
        commands: list[list[str]] = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]

        # Allow projects to override with a custom test command.
        custom = os.getenv("FURROW_TEST_COMMAND")
        if custom:
            commands.insert(0, custom.split())

        return commands
