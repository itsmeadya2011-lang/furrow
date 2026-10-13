from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from .logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.logger = get_logger(__name__)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.logger.info("Starting test execution", goal=goal, num_tasks=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            self.logger.error("Error during test execution", exc_info=True)
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            self.logger.info("Test execution completed", passed=passed, summary=response)
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        self.logger.debug("Looking for test runner")
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
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                self.logger.debug(f"Test runner not found: {' '.join(cmd)}")
                continue
            except Exception as e:
                self.logger.debug(f"Error running test command {' '.join(cmd)}: {e}")
                continue
        return "No test runner found."