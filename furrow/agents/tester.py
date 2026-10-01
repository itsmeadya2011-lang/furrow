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
    """Agent that runs tests and evaluates whether the goal is met."""

    __test__ = False  # Prevent pytest from collecting this as a test class.

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.exception("test_runner_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\nTasks completed:\n"
            f"{self._format_tasks(tasks)}\n\nTest output:\n{test_output}\n"
        )
        response = await self.client.complete(
            prompt, model=self.client.settings.tester_model
        )
        return self._parse_test_result(response)

    @staticmethod
    def _format_tasks(tasks: list[TaskModel]) -> str:
        lines = []
        for t in tasks:
            lines.append(f"  [{t.id}] {t.description} — {t.status}")
        return "\n".join(lines)

    @staticmethod
    def _parse_test_result(response: str) -> TestResult:
        """Parse the LLM's JSON test verdict, falling back gracefully."""
        cleaned = response.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            if first_nl != -1:
                cleaned = cleaned[first_nl + 1:]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return TestResult(**data)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # Fallback heuristic.
        passed = "passed" in response.lower() and "fail" not in response.lower()
        return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        """Try common test runners and return combined stdout+stderr."""
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
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=120
                    )
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", cmd=cmd)
                    continue
            except FileNotFoundError:
                continue
            except OSError as e:
                logger.warning("test_runner_error", cmd=cmd, error=str(e))
                continue
        return "No test runner found."
