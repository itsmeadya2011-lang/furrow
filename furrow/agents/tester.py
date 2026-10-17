from __future__ import annotations

import asyncio
import json
import sys
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - intentionally catch all test-runner failures
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        """Run pytest, ruff, and mypy checks and aggregate their output."""
        parts: list[str] = []

        # --- pytest ---
        pytest_output = await self._run_command(
            [sys.executable, "-m", "pytest", "-q"],
            ["pytest", "-q"],
            timeout=120,
        )
        parts.append(
            f"=== pytest ===\n{pytest_output if pytest_output else 'No pytest runner found.'}"
        )

        # --- ruff lint ---
        ruff_output = await self._run_command(
            [sys.executable, "-m", "ruff", "check", "."],
            ["ruff", "check", "."],
            timeout=60,
        )
        if ruff_output:
            parts.append(f"=== ruff ===\n{ruff_output}")

        # --- mypy type-check ---
        mypy_output = await self._run_command(
            [sys.executable, "-m", "mypy", "."],
            ["mypy", "."],
            timeout=60,
        )
        if mypy_output:
            parts.append(f"=== mypy ===\n{mypy_output}")

        return "\n\n".join(parts) if parts else "No test/lint runners found."

    @staticmethod
    async def _run_command(primary: list[str], fallback: list[str], timeout: int) -> str:
        """Run a command, trying primary then fallback. Returns combined stdout+stderr."""
        for cmd in (primary, fallback):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except Exception:  # noqa: BLE001, S112 - fallback command on any error
                continue
        return ""
