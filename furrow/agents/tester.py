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
        # Try to detect test command from common files
        test_commands = []
        if os.path.exists("pyproject.toml"):
            test_commands.append(["pytest", "-v"])
            test_commands.append(["python", "-m", "pytest", "-v"])
        if os.path.exists("package.json"):
            test_commands.append(["npm", "test", "--", "--silent"])
            test_commands.append(["pnpm", "test", "--", "--silent"])
            test_commands.append(["yarn", "test", "--silent"])
        if os.path.exists("Cargo.toml"):
            test_commands.append(["cargo", "test", "-q"])
        if os.path.exists("go.mod"):
            test_commands.append(["go", "test", "./..."])
        if os.path.exists("pytest.ini") or os.path.exists("setup.cfg"):
            test_commands.append(["pytest", "-v"])
            test_commands.append(["python", "-m", "pytest", "-v"])
        if os.path.exists("tox.ini"):
            test_commands.append(["pytest", "-v"])

        # Default fallbacks
        if not test_commands:
            test_commands = [
                ["pytest", "-v"],
                ["python", "-m", "pytest", "-v"],
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]

        for cmd in test_commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
                    output = stdout.decode() + stderr.decode()
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
