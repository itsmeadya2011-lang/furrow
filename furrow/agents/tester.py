from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if not test_output.strip():
            test_output = "No test output captured. Tests may not exist or did not run."

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = self.settings.workspace
        timeout = self.settings.test_timeout
        candidates = [
            (["pytest", "-q"], workspace),
            (["python", "-m", "pytest", "-q"], workspace),
            (["npm", "test", "--", "--silent"], workspace),
            (["pnpm", "test", "--", "--silent"], workspace),
            (["yarn", "test", "--silent"], workspace),
            (["cargo", "test", "-q"], workspace),
            (["go", "test", "./..."], workspace),
        ]
        for cmd, cwd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await proc.wait()
                    except ProcessLookupError:
                        pass
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found or all test runners failed silently."
