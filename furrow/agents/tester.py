from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT, _extract_json
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info(
            "tester_start",
            goal=goal,
            task_count=len(tasks),
            model=self.client.settings.tester_model,
        )
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.exception(
                "tester_run_failure",
                goal=goal,
                error=str(e),
            )
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = _extract_json(response)
            result = TestResult(**data)
            logger.info("tester_complete", passed=result.passed)
            return result
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "tester_json_parse_fallback",
                model=self.client.settings.tester_model,
                response_preview=response[:200],
            )
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
