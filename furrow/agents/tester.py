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


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("starting tests", goal=goal)
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.exception("error running tests", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        logger.info("sending prompt to LLM", prompt=prompt)
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        logger.info("received LLM response", response=response)
        try:
            data = json.loads(response)
            logger.info("LLM response parsed successfully")
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            logger.warning("failed to parse LLM response as JSON", response=response)
            result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])
        logger.info("test result", passed=result.passed, summary=result.summary)
        return result

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
        for cmd in candidates:
            logger.info("trying test runner", command=cmd)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    logger.info("test runner succeeded", command=cmd)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    logger.warning("test runner timed out", command=cmd)
                    proc.kill()
                    continue
            except FileNotFoundError:
                logger.warning("test runner not found", command=cmd)
                continue
        logger.warning("no test runner found")
        return "No test runner found."
