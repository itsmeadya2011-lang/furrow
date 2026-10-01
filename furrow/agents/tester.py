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
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        lint_output = ""

        try:
            test_output = await self._run_tests()
        except RuntimeError as e:
            return TestResult(
                passed=False,
                summary=f"Test execution error: {e}",
                failures=[str(e)],
            )
        except OSError as e:
            return TestResult(
                passed=False,
                summary=f"OS error during test execution: {e}",
                failures=[str(e)],
            )

        # Run linter if available
        lint_output = await self._run_lint()

        combined_output = test_output
        if lint_output:
            combined_output += "\n--- Lint Output ---\n" + lint_output

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{combined_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q", "--tb=short"],
            ["python", "-m", "pytest", "-q", "--tb=short"],
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
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", command=" ".join(cmd))
                    continue
            except FileNotFoundError:
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."

    async def _run_lint(self) -> str:
        """Run the linter if one is configured for the project."""
        lint_commands = [
            ["ruff", "check", "."],
            ["python", "-m", "ruff", "check", "."],
            ["flake8", "."],
            ["python", "-m", "flake8", "."],
        ]
        for cmd in lint_commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        logger.info("lint_ran", command=" ".join(cmd))
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
        return ""
