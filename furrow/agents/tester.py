from __future__ import annotations

import asyncio
import json
import subprocess
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self._settings = settings or default_settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("tester.start", goal=goal[:100], tasks=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            logger.error("tester.run_tests_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        tester_model = getattr(self.client, 'settings', self._settings).tester_model
        response = await self.client.complete(prompt, model=tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester.completed", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            result = TestResult(passed=passed, summary=response, failures=[] if passed else ["Failed to parse test result"])
            logger.warning("tester.parse_failed", passed=passed)
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
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    logger.debug("tester.test_run", cmd=cmd[0], exit_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("tester.test_timeout", cmd=cmd[0])
                    continue
            except FileNotFoundError:
                logger.debug("tester.test_runner_not_found", cmd=cmd[0], error="not found")
                continue
            except subprocess.SubprocessError as e:
                logger.debug("tester.test_runner_error", cmd=cmd[0], error=str(e))
                continue
        logger.warning("tester.no_test_runner")
        return "No test runner found."
