from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, ClassVar

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class TesterAgent:
    DEFAULT_TEST_COMMANDS: ClassVar[list[list[str]]] = [
        ["pytest", "-q"],
        ["python", "-m", "pytest", "-q"],
        ["npm", "test", "--", "--silent"],
        ["pnpm", "test", "--", "--silent"],
        ["yarn", "test", "--silent"],
        ["cargo", "test", "-q"],
        ["go", "test", "./..."],
    ]

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings
        self.test_timeout: int = self.settings.test_timeout
        self.test_command: str | None = self.settings.test_command

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("testing_started", goal=goal, task_count=len(tasks))

        try:
            test_output = await self._run_tests()
        except RuntimeError as e:
            logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("testing_succeeded", goal=goal, passed=result.passed)
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("test_result_parse_failed", error=str(e))
            passed = "passed" in response.lower()
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        if self.test_command:
            return await self._run_shell_command(self.test_command)

        candidates = self.DEFAULT_TEST_COMMANDS
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except FileNotFoundError:
                logger.debug("test_command_not_found", command=cmd[0])
                continue

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.test_timeout
                )
                logger.debug("test_command_succeeded", command=cmd[0])
                return stdout.decode() + stderr.decode()
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning("test_command_timeout", command=cmd[0], timeout=self.test_timeout)
                continue

        logger.error("no_test_runner_found", candidates=[c[0] for c in candidates])
        raise RuntimeError(
            "No test runner found. Install pytest or npm in your environment, "
            "or set FURROW_TEST_COMMAND to a custom command."
        )

    async def _run_shell_command(self, command: str) -> str:
        logger.debug("running_custom_test_command", command=command)
        proc = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.test_timeout
            )
            return stdout.decode() + stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(
                f"Test command timed out after {self.test_timeout}s: {command}"
            )
