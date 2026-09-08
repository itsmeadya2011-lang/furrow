from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


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
        commands = self._detect_test_commands()
        if not commands:
            return "No test configuration found in project. Skipping tests."

        for cmd in commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
                )
                try:
                    output, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
                    return output.decode()
                except asyncio.TimeoutError:
                    logger.debug("Test command timed out: %s", cmd)
                    proc.kill()
                    await proc.wait()
                    continue
            except FileNotFoundError:
                logger.debug("Test runner not found on PATH: %s", cmd[0])
                continue
        return "No test runner found."

    def _detect_test_commands(self) -> list[list[str]]:
        commands: list[list[str]] = []

        if os.path.exists("pyproject.toml") or os.path.exists("setup.py") or os.path.exists("requirements.txt"):
            commands.append(["pytest", "-q"])
            commands.append(["python", "-m", "pytest", "-q"])

        if os.path.exists("package.json"):
            with open("package.json") as f:
                pkg = json.load(f)
            if isinstance(pkg, dict) and isinstance(pkg.get("scripts"), dict) and "test" in pkg["scripts"]:
                commands.append(["npm", "test", "--", "--silent"])
                commands.append(["pnpm", "test", "--", "--silent"])
                commands.append(["yarn", "test", "--silent"])

        if os.path.exists("Cargo.toml"):
            commands.append(["cargo", "test", "-q"])

        if os.path.exists("go.mod"):
            commands.append(["go", "test", "./..."])

        return commands
