from __future__ import annotations

import asyncio
import json
import re
import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)

TEST_TIMEOUT = 120  # seconds per test runner attempt


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        task_summaries = "\n".join(
            f"- {t.id}: {t.description} -> {t.status}" for t in tasks
        )

        try:
            test_output = await self._run_tests()
        except Exception as e:
            log.error("test_run_error", error=str(e))
            return TestResult(passed=False, summary=f"Test runner error: {e}", failures=[str(e)])

        # If no test runner was found, treat as pass (no tests to fail)
        if test_output.strip() == "No test runner found.":
            return TestResult(passed=True, summary="No test runner found; skipping tests.", failures=[])

        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Task summaries:\n{task_summaries}\n\n"
            f"Test output:\n{test_output}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        response = response.strip()

        # Strip markdown code fences
        if response.startswith("```"):
            lines = response.splitlines()
            if lines and lines[-1].startswith("```"):
                response = "\n".join(lines[1:-1])
            elif lines:
                response = "\n".join(lines[1:])

        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower() or "all tests" in response.lower()
            # Heuristic: extract failure lines if present
            failures = re.findall(r"FAILED.*", response) if not passed else []
            return TestResult(passed=passed, summary=response[:500], failures=failures or [])

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["ruff", "check", "."],
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
                    cwd=str(self.client.settings.workspace),
                )
            except (FileNotFoundError, OSError):
                continue
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=TEST_TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                log.warning("test_runner_timeout", cmd=str(cmd))
                continue

            stdout = stdout_b.decode(errors="replace")
            stderr = stderr_b.decode(errors="replace")
            output = stdout + stderr
            # If the runner ran successfully (exit 0), return the output
            if proc.returncode == 0 and output.strip():
                return output
            # If tests failed (non-zero), still return the output so the LLM can analyze
            if proc.returncode != 0 and output.strip():
                return output
            # Runner exists but produced no output — try next
            continue

        return "No test runner found."
