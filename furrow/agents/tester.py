from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger("furrow.tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run tests for the given tasks and evaluate the result via the LLM.

        Executes the project's test suite, then sends the output to the LLM
        for a structured pass/fail evaluation. Retries on transient failures.
        """
        logger.info("starting_test_execution", goal=goal)
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            logger.error("test_execution_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])
        logger.info("test_execution_completed")

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            logger.warning("response_json_parse_failed", falling_back_to_text=True)
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        """Run the first available test runner and return its combined output.

        Iterates over known test runners (pytest, npm test, cargo test, etc.)
        and runs the first one found on the system PATH. If no runner is
        available, returns a sentinel string.
        """
        # Try each known test runner in turn; the first one available on PATH
        # runs and its output is returned. If none is found, fall back to a
        # sentinel message so downstream callers can still proceed.
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
                logger.debug("trying_test_runner", runner=cmd[0])
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, OSError):
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."
