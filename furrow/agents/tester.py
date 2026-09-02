from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    """Runs the project's test suite and evaluates results via an LLM.

    Tries multiple common test runners (pytest, npm, pnpm, yarn, cargo, go).
    If no runner is found, falls back to an LLM analysis of the situation.
    """

    TEST_COMMANDS: list[list[str]] = [
        ["pytest", "-q"],
        ["python", "-m", "pytest", "-q"],
        ["python", "-m", "unittest", "-v"],
        ["npm", "test", "--", "--silent"],
        ["pnpm", "test", "--", "--silent"],
        ["yarn", "test", "--silent"],
        ["cargo", "test", "-q"],
        ["go", "test", "./..."],
        ["bun", "test", "--silent"],
    ]

    TEST_TIMEOUT: int = 120

    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run tests and return a structured TestResult.

        If running the test suite fails (no runner found, or command errors),
        that is treated as a test failure rather than crashing.
        """
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(
                passed=False,
                summary=f"Failed to run tests: {e}",
                failures=[f"Test execution error: {e}"],
            )

        if "No test runner found" in test_output:
            # No test infrastructure — use LLM to evaluate based on changes
            return await self._evaluate_without_tests(goal, tasks)

        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n"
            f"Tasks completed:\n"
            + "\n".join(f"  - {t.description}: {t.status}" for t in tasks)
            + f"\n\nTest output:\n{test_output}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        # Try parsing JSON; retry once on failure (LLM may need a nudge).
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            # Fallback: retry with a stricter instruction
            retry_prompt = (
                f"{TESTER_PROMPT}\n\n"
                f"IMPORTANT: You must return ONLY valid JSON. Do not include any text before or after.\n\n"
                f"Goal: {goal}\n"
                f"Tasks completed:\n"
                + "\n".join(f"  - {t.description}: {t.status}" for t in tasks)
                + f"\n\nTest output:\n{test_output}\n"
            )
            response = await self.client.complete(
                retry_prompt, model=self.client.settings.tester_model
            )
            try:
                data = json.loads(response)
            except (json.JSONDecodeError, ValueError):
                # Last-resort fallback: heuristic
                return TestResult(
                    passed="passed" in response.lower() and "fail" not in response.lower(),
                    summary=response[:500],
                    failures=[],
                )

        try:
            return TestResult(**data)
        except (TypeError, ValueError) as e:
            # LLM returned valid JSON but wrong shape
            passed = data.get("passed") if isinstance(data, dict) else None
            if passed is None:
                passed = "passed" in response.lower() and "fail" not in response.lower()
            return TestResult(
                passed=passed,
                summary=data.get("summary", response[:500]) if isinstance(data, dict) else response[:500],
                failures=data.get("failures", []) if isinstance(data, dict) else [],
            )

    async def _evaluate_without_tests(
        self, goal: str, tasks: list[TaskModel]
    ) -> TestResult:
        """When no test runner is available, use LLM to assess completion."""
        prompt = (
            f"Goal: {goal}\n\n"
            f"The project does not have an automated test runner. "
            f"Based on the completed tasks, assess whether the goal appears to be met.\n\n"
            f"Tasks:\n"
            + "\n".join(f"  - {t.description}: {t.status} — {t.result or ''}" for t in tasks)
            + "\n\nReturn JSON with 'passed', 'summary', 'failures'."
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower() and "fail" not in response.lower(),
                summary=response[:500],
                failures=[],
            )

    async def _run_tests(self) -> str:
        """Attempt to run the test suite using any available test runner.

        Returns the combined stdout+stderr output. If no runner is found,
        returns a message indicating that.
        """
        for cmd in self.TEST_COMMANDS:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self.TEST_TIMEOUT
                    )
                    result = stdout.decode() + stderr.decode()
                    if result.strip():
                        return result
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, PermissionError, OSError):
                continue
            except Exception:
                continue
        return "No test runner found."
