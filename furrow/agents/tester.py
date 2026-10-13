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

logger = structlog.get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("tester_start", task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except (RuntimeError, OSError, ValueError, asyncio.TimeoutError) as e:
            logger.error("tester_run_tests_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        logger.debug("tester_test_output", output_length=len(test_output))
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester_done", passed=result.passed, summary=result.summary, failure_count=len(result.failures))
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("tester_parse_failed", error=str(e), response=response[:500])
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
                logger.debug("tester_try_runner", cmd=cmd[0])
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    logger.debug("tester_runner_done", cmd=cmd[0], returncode=proc.returncode, output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    logger.warning("tester_runner_timeout", cmd=cmd[0])
                    proc.kill()
                    continue
            except (FileNotFoundError, RuntimeError, OSError) as e:
                logger.debug("tester_runner_not_found", cmd=cmd[0], error=str(e))
                continue
        logger.warning("tester_no_runner_found")
        return "No test runner found."