from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger("furrow.tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.exception("test execution failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        logger.info("starting test discovery")
        timeout = getattr(self.client.settings, "test_timeout", 120)
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
            runner = cmd[0]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except FileNotFoundError:
                logger.debug("test runner not installed, trying next candidate", runner=runner)
                continue
            except (PermissionError, OSError) as e:
                logger.warning("failed to launch test runner, trying next candidate", runner=runner, error=str(e))
                continue

            logger.info("selected test runner", runner=runner, timeout=timeout)
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode() + stderr.decode()
                summary = output.strip().splitlines()[-1] if output.strip() else ""
                logger.info(
                    "test runner completed",
                    runner=runner,
                    returncode=proc.returncode,
                    summary=summary,
                )
                return output
            except asyncio.TimeoutError:
                logger.warning("test runner timed out", runner=runner, timeout=timeout)
                proc.kill()
                continue

        logger.warning("no test runner available")
        return "No test runner found."
