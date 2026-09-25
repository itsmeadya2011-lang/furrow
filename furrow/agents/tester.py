from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        lint_output = ""
        typecheck_output = ""
        
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])
        
        try:
            lint_output = await self._run_lint()
        except Exception:
            pass
            
        try:
            typecheck_output = await self._run_typecheck()
        except Exception:
            pass

        prompt = f"""{TESTER_PROMPT}

Goal: {goal}

Test output:
{test_output or "No tests found or test runner not available."}

Lint output:
{lint_output or "No lint issues."}

Typecheck output:
{typecheck_output or "No typecheck issues."}

Tasks completed:
{json.dumps([{"id": t.id, "description": t.description, "result": t.result} for t in tasks], indent=2)}

Return JSON only with this shape:
{{
  "passed": true/false,
  "summary": "Brief summary",
  "failures": ["failure1", "failure2"]
}}"""

        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback parsing
            passed = "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(passed=passed, summary=response, failures=[])

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

    async def _run_lint(self) -> str:
        candidates = [
            ["ruff", "check", "."],
            ["eslint", "."],
            ["golangci-lint", "run"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return ""

    async def _run_typecheck(self) -> str:
        candidates = [
            ["mypy", "."],
            ["tsc", "--noEmit"],
            ["go", "vet", "./..."],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return ""
