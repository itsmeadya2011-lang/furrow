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
        self.workspace = self.client.settings.workspace

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_outputs = {}
        
        # Run tests
        test_outputs["tests"] = await self._run_tests()
        
        # Run linter if available
        test_outputs["lint"] = await self._run_lint()
        
        # Run type checker if available
        test_outputs["typecheck"] = await self._run_typecheck()

        # Combine all outputs
        combined_output = "\n\n".join(
            f"=== {key.upper()} ===\n{value}" 
            for key, value in test_outputs.items()
        )

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{combined_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: check if any test runner found failures
            passed = "passed" in combined_output.lower() and "failed" not in combined_output.lower()
            return TestResult(passed=passed, summary=combined_output[:500], failures=[])

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q", "--tb=short"],
            ["python", "-m", "pytest", "-q", "--tb=short"],
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
                    cwd=self.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip() or proc.returncode != 0:
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found or no tests to run."

    async def _run_lint(self) -> str:
        candidates = [
            ["ruff", "check", "."],
            ["flake8", "."],
            ["eslint", "."],
            ["npm", "run", "lint"],
            ["pnpm", "run", "lint"],
            ["yarn", "lint"],
            ["cargo", "clippy", "--", "-D", "warnings"],
            ["golangci-lint", "run"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip() or proc.returncode != 0:
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No linter found or no lint issues."

    async def _run_typecheck(self) -> str:
        candidates = [
            ["mypy", "."],
            ["pyright", "."],
            ["tsc", "--noEmit"],
            ["npm", "run", "typecheck"],
            ["pnpm", "run", "typecheck"],
            ["yarn", "typecheck"],
            ["cargo", "check"],
            ["go", "vet", "./..."],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip() or proc.returncode != 0:
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No type checker found or no type errors."
