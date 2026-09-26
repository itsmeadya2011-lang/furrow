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

logger = structlog.get_logger(__name__)


class TesterAgent:
    __test__ = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    @property
    def _timeout(self) -> int:
        return getattr(self.client.settings, "timeout", 120)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("test_run_started", goal=goal, num_tasks=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - intentional catch for test failure reporting
            logger.error("test_execution_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        logger.info("test_run_complete", output_length=len(test_output))

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info(
                "test_result_parsed",
                passed=result.passed,
                num_failures=len(result.failures),
                summary=result.summary[:100],
            )
            return result
        except (json.JSONDecodeError, ValueError):
            logger.warning("test_result_parse_failed", response_preview=response[:200])
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["python3", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
            try:
                logger.info("trying_test_command", command=cmd)
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self._timeout
                    )
                    output = stdout.decode() + stderr.decode()
                    logger.info("test_command_succeeded", command=cmd, output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    logger.warning("test_command_timeout", command=cmd, timeout=self._timeout)
                    proc.kill()
                    continue
            except (FileNotFoundError, OSError) as e:
                logger.debug("test_command_not_found", command=cmd, error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."

    async def run_lint(self) -> str:
        logger.info("lint_run_started")
        candidates = [
            ["ruff", "check", "."],
            ["python", "-m", "ruff", "check", "."],
            ["python3", "-m", "ruff", "check", "."],
            ["flake8", "."],
        ]
        for cmd in candidates:
            try:
                logger.info("trying_lint_command", command=cmd)
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self._timeout
                    )
                    output = stdout.decode() + stderr.decode()
                    logger.info("lint_command_succeeded", command=cmd, output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    logger.warning("lint_command_timeout", command=cmd, timeout=self._timeout)
                    proc.kill()
                    continue
            except (FileNotFoundError, OSError) as e:
                logger.debug("lint_command_not_found", command=cmd, error=str(e))
                continue
        logger.warning("no_linter_found")
        return "No linter found."

    async def run_type_check(self) -> str:
        logger.info("type_check_run_started")
        candidates = [
            ["mypy", "."],
            ["python", "-m", "mypy", "."],
            ["python3", "-m", "mypy", "."],
        ]
        for cmd in candidates:
            try:
                logger.info("trying_type_check_command", command=cmd)
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self._timeout
                    )
                    output = stdout.decode() + stderr.decode()
                    logger.info(
                        "type_check_command_succeeded", command=cmd, output_length=len(output)
                    )
                    return output
                except asyncio.TimeoutError:
                    logger.warning("type_check_command_timeout", command=cmd, timeout=self._timeout)
                    proc.kill()
                    continue
            except (FileNotFoundError, OSError) as e:
                logger.debug("type_check_command_not_found", command=cmd, error=str(e))
                continue
        logger.warning("no_type_checker_found")
        return "No type checker found."
