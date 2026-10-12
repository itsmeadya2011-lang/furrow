from __future__ import annotations

import asyncio
import json
import os
from shutil import which
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
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
        timeout = getattr(self.client.settings, "request_timeout", 120)

        cmd = self._detect_runner()
        if cmd is None:
            # Fall back to trying every candidate runner in order.
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]
            for candidate in candidates:
                output = await self._run_command(candidate, timeout)
                if output is not None:
                    return output
            return "No test runner found."

        output = await self._run_command(cmd, timeout)
        if output is None:
            return "No test runner found."
        return output

    @staticmethod
    def _detect_runner() -> list[str] | None:
        """Pick a single test command based on project marker files."""
        if os.path.exists("pyproject.toml") or os.path.exists("setup.py"):
            return ["pytest", "-q"]
        if os.path.exists("package.json"):
            for runner in ("npm", "pnpm", "yarn"):
                if which(runner) is not None:
                    return [runner, "test", "--silent"]
            return None
        if os.path.exists("Cargo.toml"):
            return ["cargo", "test", "-q"]
        if os.path.exists("go.mod"):
            return ["go", "test", "./..."]
        return None

    async def _run_command(self, cmd: list[str], timeout: float) -> str | None:
        """Run a single command, returning its output with the exit code prepended.

        Returns None when the executable is missing so the caller can try the
        next candidate. Any other error is propagated to the caller.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except FileNotFoundError:
            # The runner executable isn't installed; skip to the next candidate.
            return None

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"EXIT_CODE: -1\nTimeout after {timeout}s running {' '.join(cmd)}\n"

        output = stdout.decode() + stderr.decode()
        return f"EXIT_CODE: {proc.returncode}\n{output}"