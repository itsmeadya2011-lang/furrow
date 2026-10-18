from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult, settings as default_settings
from furrow.llm import LLMClient


class TesterAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(
            prompt, model=self.client.settings.tester_model
        )
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower(),
                summary=response,
                failures=[],
            )

    async def _run_tests(self) -> str:
        workspace = self.settings.workspace
        env = dict(os.environ)

        candidates: list[list[str]] = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["python3", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]

        # Prefer pytest when the workspace looks like a Python project.
        if (workspace / "pyproject.toml").exists() or (workspace / "setup.py").exists() or (workspace / "tests").exists():
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["python3", "-m", "pytest", "-q"],
            ] + candidates

        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace),
                    env=env,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self.settings.test_timeout
                    )
                    output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
                    if proc.returncode == 0 or "No test runner found" not in output:
                        return f"Command: {' '.join(cmd)}\nExit code: {proc.returncode}\n{output}"
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return (
                        f"Command: {' '.join(cmd)}\n"
                        f"Timed out after {self.settings.test_timeout}s.\n"
                    )
            except (FileNotFoundError, OSError):
                continue
        return "No test runner found."
