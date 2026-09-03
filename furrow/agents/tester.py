from __future__ import annotations

import asyncio
import json
import os
import shutil
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
            lint_typecheck_output = await self._run_lint_and_typecheck()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        combined_output = f"--- Test Output ---\n{test_output}\n\n--- Lint & Typecheck Output ---\n{lint_typecheck_output}\n"

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nOutput:\n{combined_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

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
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."

    async def _run_lint_and_typecheck(self) -> str:
        candidates = [
            ("ruff", ["ruff", "check", "."]),
            ("mypy", ["mypy", "."]),
        ]
        results: list[str] = []
        for name, cmd in candidates:
            if shutil.which(cmd[0]) is None:
                results.append(f"{name}: not installed (skipped)")
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    results.append(f"--- {name} ---\n{output}")
                except asyncio.TimeoutError:
                    proc.kill()
                    results.append(f"--- {name} ---\nTimeout after 60 seconds")
            except (FileNotFoundError, Exception) as e:
                results.append(f"--- {name} ---\nError: {e}")
        return "\n\n".join(results) if results else "No linters/type checkers found."