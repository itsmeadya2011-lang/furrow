from __future__ import annotations

import asyncio
import json
import logging
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
            logger.error("test_execution_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        except Exception as e:
            logger.error("tester_llm_failed", error=str(e))
            return TestResult(passed=False, summary=f"Tester LLM failed: {e}", failures=[str(e)])

        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower(),
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
                    logger.info("test_runner_found", cmd=" ".join(cmd), output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    logger.warning("test_runner_timeout", cmd=" ".join(cmd))
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        logger.info("no_test_runner_found")
        return "No test runner found."
