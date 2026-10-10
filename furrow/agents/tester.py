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
        try:
            test_output = await self._run_tests()
        except FileNotFoundError:
            # No test runner was found on the system.
            logger.warning("no test runner found")
            return TestResult(
                passed=False,
                summary="No test runner found.",
                failures=["No supported test runner (pytest, npm, pnpm, yarn, cargo, go) is installed."],
            )
        except asyncio.TimeoutError:
            logger.warning("test run timed out")
            return TestResult(
                passed=False,
                summary="Test run timed out.",
                failures=["The test runner exceeded the 120s timeout."],
            )
        except Exception as e:
            # Unexpected error: log it and surface a clear failure rather than
            # silently swallowing the exception.
            logger.exception("unexpected error while running tests", error=e)
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        logger.info("tests run", output_length=len(test_output))

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            logger.warning("failed to parse LLM response as JSON", response=response)
            return self._fallback_test_result(response)

    @staticmethod
    def _fallback_test_result(response: str) -> TestResult:
        """Best-effort interpretation of a non-JSON LLM response."""
        lowered = response.lower()
        # Look for explicit failure indicators first; default to passed only
        # when nothing suggests a failure.
        if any(token in lowered for token in ("fail", "error", "not passed", "not pass")):
            return TestResult(passed=False, summary=response, failures=[response])
        if "pass" in lowered:
            return TestResult(passed=True, summary=response, failures=[])
        # No clear signal either way: assume failure to be conservative.
        return TestResult(passed=False, summary=response, failures=[response])

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
            except FileNotFoundError:
                # Command not found — try the next candidate runner.
                logger.debug("test runner not available", command=cmd[0])
                continue

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode() + stderr.decode()
                logger.info("test runner succeeded", command=cmd[0])
                return output
            except asyncio.TimeoutError:
                proc.kill()
                # Timeout is not a "command not found" situation, but we cannot
                # recover from it for this candidate either; record it and move
                # on to the next runner.
                logger.warning("test runner timed out", command=cmd[0])
                continue
        return "No test runner found."