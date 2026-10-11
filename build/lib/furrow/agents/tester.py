from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    pass


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.workspace = Path(self.settings.workspace)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # First, let LLM analyze the test output
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])

        # If tests failed, try to fix them
        if not result.passed and result.failures:
            result = await self._try_fix_failures(goal, tasks, result, test_output)

        return result

    async def _run_tests(self) -> str:
        # Use custom test command if configured
        if self.settings.test_command:
            cmd = shlex.split(self.settings.test_command)
            return await self._run_command(cmd)

        # Otherwise try common test runners
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
                output = await self._run_command(cmd)
                if output.strip():
                    return output
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."

    async def _run_command(self, cmd: list[str], timeout: int = 120) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout.decode() + stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(cmd)}")

    async def _try_fix_failures(
        self, goal: str, tasks: list[TaskModel], result: TestResult, test_output: str
    ) -> TestResult:
        """Attempt to fix test failures by spawning a worker for each failure."""
        # For now, just return the result with failures
        # In a more advanced version, we could spawn workers to fix each failure
        return result
