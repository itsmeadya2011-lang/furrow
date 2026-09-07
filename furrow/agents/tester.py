from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.logger = get_logger("furrow.tester")

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            self.logger.error("test_execution_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            self.logger.info("test_result", passed=result.passed, failures=len(result.failures))
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            self.logger.info("test_result_fallback", passed=passed)
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = self._detect_test_commands()
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
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."

    def _detect_test_commands(self) -> list[list[str]]:
        workspace = self.client.settings.workspace
        if (workspace / "package.json").exists():
            return [["npm", "test", "--", "--silent"], ["pnpm", "test", "--", "--silent"], ["yarn", "test", "--silent"]]
        if (workspace / "pyproject.toml").exists() or (workspace / "setup.py").exists():
            return [["python", "-m", "pytest", "-q"], ["pytest", "-q"]]
        if (workspace / "Cargo.toml").exists():
            return [["cargo", "test", "-q"]]
        if (workspace / "go.mod").exists():
            return [["go", "test", "./..."]]
        return [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
