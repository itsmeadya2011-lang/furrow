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


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        test_candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]

        lint_candidates = [
            ["ruff", "check", "."],
            ["python", "-m", "ruff", "check", "."],
            ["mypy", "."],
            ["python", "-m", "mypy", "."],
            ["npm", "run", "lint"],
            ["pnpm", "lint"],
            ["yarn", "lint"],
            ["cargo", "clippy"],
            ["go", "vet", "./..."],
        ]

        combined_output = []

        # Run test runners first
        for cmd in test_candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    combined_output.append(stdout.decode() + stderr.decode())
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except Exception:
                continue

        # Run linting/type checking commands after test runners
        for cmd in lint_candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    combined_output.append(stdout.decode() + stderr.decode())
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except Exception:
                continue

        if not combined_output:
            return "No test runner found."

        return "\n".join(combined_output)
