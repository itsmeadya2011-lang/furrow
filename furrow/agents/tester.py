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

log = structlog.get_logger(__name__)


class TesterAgent:
    """Runs the project's test suite and evaluates pass/fail via the LLM."""

    TEST_RUNNERS: list[list[str]] = [
        ["pytest", "-q", "--tb=short"],
        ["python", "-m", "pytest", "-q", "--tb=short"],
        ["npm", "test", "--silent"],
        ["pnpm", "test", "--silent"],
        ["yarn", "test", "--silent"],
        ["cargo", "test", "-q"],
        ["go", "test", "./..."],
    ]

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        # Detect test framework early
        test_output = await self._run_tests()
        log.debug("Test output collected", length=len(test_output))

        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n"
            f"Completed tasks: {len([t for t in tasks if t.status == 'completed'])}\n"
            f"Test output:\n{test_output}\n"
        )

        response = await self.client.complete(
            prompt,
            model=self.client.settings.tester_model,
        )

        # Robust JSON parsing (handles markdown-wrapped JSON)
        stripped = response.strip()
        if stripped.startswith("```json"):
            stripped = stripped[len("```json") :]
        elif stripped.startswith("```"):
            stripped = stripped[len("```") :]
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]

        try:
            data = json.loads(stripped)
            result = TestResult(**data)
            log.info("Test result parsed", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("Failed to parse test result JSON, using fallback", error=str(e))
            passed = "passed" in response.lower() and "fail" not in response.lower()
            return TestResult(
                passed=passed,
                summary=response[:300],
                failures=[] if passed else [response[:300]],
            )

    async def _run_tests(self) -> str:
        """Try each test runner until one is found and runs."""
        for cmd in self.TEST_RUNNERS:
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
                    output = stdout.decode() + stderr.decode()
                    log.debug("Test runner found", command=cmd[0])
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    log.warning("Test runner timed out", command=cmd[0])
                    continue
            except (FileNotFoundError, OSError):
                continue
        log.warning("No test runner found")
        return "No test runner found."
