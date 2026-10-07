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
        output_parts = []
        
        # Run tests
        test_output = await self._run_command([
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ])
        if test_output:
            output_parts.append(f"=== Tests ===\n{test_output}")
        
        # Run linting
        lint_output = await self._run_command([
            ["ruff", "check", "."],
            ["flake8", "."],
            ["eslint", "."],
            ["golangci-lint", "run"],
        ])
        if lint_output:
            output_parts.append(f"=== Lint ===\n{lint_output}")
        
        # Run type checking
        type_output = await self._run_command([
            ["mypy", "."],
            ["pyright", "."],
            ["tsc", "--noEmit"],
            ["go", "vet", "./..."],
        ])
        if type_output:
            output_parts.append(f"=== Type Check ===\n{type_output}")
        
        return "\n\n".join(output_parts) if output_parts else "No test/lint/type check runners found."

    async def _run_command(self, candidates: list[list[str]]) -> str:
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    result = stdout.decode() + stderr.decode()
                    if result.strip():
                        return result
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return ""
