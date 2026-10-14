from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import LintResult, TaskModel, TestResult, settings
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
        except OSError as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def run_lint(self, tasks: list[TaskModel]) -> LintResult:
        lint_output = ""
        try:
            lint_output = await self._run_lint()
        except OSError as e:
            return LintResult(passed=False, summary=str(e), issues=[str(e)])

        prompt = f"""You are a linter agent in an autonomous coding system called Furrow.

Given the lint output, determine if linting passed or failed.

Return JSON only (no markdown) with this shape:
{{
  "passed": true,
  "summary": "All lint checks passed",
  "issues": []
}}

If lint failed, list each issue in the issues array.

Lint output:
{lint_output}
"""
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return LintResult(**data)
        except (json.JSONDecodeError, ValueError):
            return LintResult(passed="passed" in response.lower(), summary=response, issues=[])

    async def _run_tests(self) -> str:
        for cmd in settings.test_commands:
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
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return "No test runner found."

    async def _run_lint(self) -> str:
        results = []
        for cmd in settings.lint_commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        results.append(f"$ {' '.join(cmd)}\n{output}")
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                continue
        
        for cmd in settings.type_check_commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        results.append(f"$ {' '.join(cmd)}\n{output}")
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                continue
        
        return "\n\n".join(results) if results else "No lint/type check tools found."
