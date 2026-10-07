from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)

TEST_COMMAND_CANDIDATES = [
    ["pytest", "-q"],
    ["python", "-m", "pytest", "-q"],
    ["npm", "test", "--", "--silent"],
    ["pnpm", "test", "--", "--silent"],
    ["yarn", "test", "--silent"],
    ["cargo", "test", "-q"],
    ["go", "test", "./..."],
]


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("testing_start", goal=goal, test_command_candidates=TEST_COMMAND_CANDIDATES)
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.info("testing_complete", passed=False, summary=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])
        logger.info("testing_complete", passed=result.passed, summary=result.summary)
        return result

    async def _run_tests(self) -> str:
        for cmd in TEST_COMMAND_CANDIDATES:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    logger.info("test_runner_found", command=" ".join(cmd))
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        logger.info("test_runner_not_found")
        return "No test runner found."
