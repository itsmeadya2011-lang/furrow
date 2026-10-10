from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_LLM_RETRIES = 3


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)], raw_output=test_output)

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            response = await self._complete_with_retries(prompt)
        except (asyncio.TimeoutError, Exception):
            # LLM call failed entirely — fall back to a simple string-based check
            passed = self._simple_check(test_output)
            return TestResult(
                passed=passed,
                summary="LLM analysis failed after retries; used simple string-based check",
                failures=[] if passed else ["LLM analysis unavailable"],
                raw_output=test_output,
            )

        try:
            data = json.loads(response)
            data.setdefault("raw_output", test_output)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            return TestResult(passed=passed, summary=response, failures=[], raw_output=test_output)

    async def _complete_with_retries(self, prompt: str, timeout: float = 120) -> str:
        """Call the LLM with a per-attempt timeout, retrying transient failures."""
        last_error: Exception | None = None
        for attempt in range(MAX_LLM_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self.client.complete(prompt, model=self.client.settings.tester_model),
                    timeout=timeout,
                )
            except Exception as e:
                last_error = e
        raise RuntimeError(
            f"LLM call failed after {MAX_LLM_RETRIES + 1} attempts: {last_error}"
        )

    @staticmethod
    def _simple_check(output: str) -> bool:
        """Fallback pass/fail check based on raw output strings."""
        if not output:
            return False
        lower = output.lower()
        failure_markers = ("failed", "error", "failure", "no tests found", "traceback")
        return not any(m in lower for m in failure_markers)

    async def _run_tests(self) -> str:
        # Test runners
        runner_candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]

        # Linters / type checkers / formatters
        lint_candidates = [
            ("ruff check .", "ruff (linter)"),
            ("mypy .", "mypy (type checker)"),
            ("eslint .", "eslint (linter)"),
            ("tsc --noEmit", "tsc (type checker)"),
            ("ruff format --check .", "ruff format (formatter)"),
        ]

        outputs: list[str] = []

        # Run test runners
        for cmd in runner_candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    combined = (stdout.decode() + stderr.decode()).strip()
                    if combined:
                        outputs.append(f"=== Test runner: {' '.join(cmd)} ===\n{combined}")
                        break
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue

        # Run linters / type checkers / formatters
        for cmd_str, label in lint_candidates:
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd_str, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    combined = (stdout.decode() + stderr.decode()).strip()
                    if combined:
                        outputs.append(f"=== {label} ===\n{combined}")
                except asyncio.TimeoutError:
                    proc.kill()
            except (FileNotFoundError, Exception):
                continue

        if not outputs:
            return "No test runner or linter found."
        return "\n\n".join(outputs)
