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
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("tester.test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        logger.debug("tester.test_output", output=test_output[:2000])

        prompt = TESTER_PROMPT.format(goal=goal, test_output=test_output)
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        logger.debug("tester.llm_response", response=response[:2000])

        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info(
                "tester.result",
                passed=result.passed,
                summary=result.summary,
                failure_count=len(result.failures),
            )
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("tester.parse_failed", error=str(e), response=response[:500])
            passed = "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(
                passed=passed,
                summary=response,
                failures=[],
            )

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
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        logger.debug("tester.runner_found", command=" ".join(cmd))
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
