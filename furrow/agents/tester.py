from __future__ import annotations

import asyncio
import json
import sys
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_execution_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = TESTER_PROMPT.format(goal=goal, test_output=test_output)
        response = await self.client.complete(
            prompt, system=TESTER_SYSTEM, model=self.client.settings.tester_model
        )
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("test_result_parse_fallback", error=str(e))
            return TestResult(
                passed="passed" in response.lower(),
                summary=response,
                failures=[],
            )

    async def _run_tests(self) -> str:
        # Use sys.executable for Python-based runners to avoid PATH issues.
        python = sys.executable
        candidates = [
            [python, "-m", "pytest", "-q"],
            ["npx", "jest", "--silent", "--passWithNoTests"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", command=" ".join(cmd))
                    continue
            except (FileNotFoundError, OSError):
                continue
        return "No test runner found."
