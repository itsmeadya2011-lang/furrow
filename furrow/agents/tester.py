from __future__ import annotations

import asyncio
import json
import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("tester_run_start", goal=goal)
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("tester_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester_run_complete", passed=result.passed)
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            logger.info("tester_run_complete", passed=passed)
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        test_runner_candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        logger.info("test_execution_start")
        test_output = "No test runner found."
        for cmd in test_runner_candidates:
            output = await self._run_command(cmd)
            if output is not None:
                test_output = output
                logger.info("test_runner_succeeded", command=" ".join(cmd))
                break
            logger.debug("test_runner_skipped", command=" ".join(cmd))

        lint_candidates = [
            ("ruff check .", ["ruff", "check", "."]),
            ("mypy furrow/", ["mypy", "furrow/"]),
        ]
        for label, cmd in lint_candidates:
            output = await self._run_command(cmd)
            if output is not None:
                logger.info("lint_complete", command=label)
                test_output += f"\n--- {label} ---\n{output}"
                if "No tests" in output or output.strip() == "":
                    logger.debug("lint_no_issues", command=label)

        if test_output.startswith("No test runner found"):
            logger.warning("no_test_runner_found")
        return test_output

    async def _run_command(self, cmd: list[str], timeout: float = 120) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return (stdout.decode() + stderr.decode()) if stdout or stderr else ""
            except asyncio.TimeoutError:
                proc.kill()
                return None
        except (FileNotFoundError, OSError):
            return None
