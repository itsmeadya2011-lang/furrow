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


NO_TEST_RUNNER_SENTINEL = "NO_TEST_RUNNER_FOUND"
TEST_RUNNER_TIMEOUT_SENTINEL = "TEST_RUNNER_TIMEOUT"


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if test_output.startswith(NO_TEST_RUNNER_SENTINEL):
            return TestResult(passed=False, summary=test_output, failures=[test_output])
        if test_output.startswith(TEST_RUNNER_TIMEOUT_SENTINEL):
            return TestResult(passed=False, summary=test_output, failures=[test_output])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        tried: list[str] = []
        for cmd in candidates:
            runner = cmd[0]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except FileNotFoundError:
                tried.append(runner)
                continue
            tried.append(runner)
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                return stdout.decode() + stderr.decode()
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"{TEST_RUNNER_TIMEOUT_SENTINEL}: runner '{runner}' did not finish within 120s"
        return f"{NO_TEST_RUNNER_SENTINEL}: tried {', '.join(tried) if tried else 'none'}"
