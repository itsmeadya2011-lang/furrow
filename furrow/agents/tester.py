from __future__ import annotations

import asyncio
import json
from pathlib import Path
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
        except FileNotFoundError as e:
            return TestResult(passed=False, summary=f"No test runner found: {e}", failures=[str(e)])
        except asyncio.TimeoutError:
            return TestResult(passed=False, summary="Tests timed out after 120s", failures=["TimeoutError"])

        prompt = f"{TESTER_PROMPT.format(test_output=test_output)}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = getattr(self.client.settings, "workspace", Path.cwd())

        if (workspace / "pyproject.toml").exists():
            candidates = [
                ["python", "-m", "pytest", "-q"],
                ["pytest", "-q"],
                ["python", "-m", "ruff", "check"],
                ["python", "-m", "mypy"],
            ]
        elif (workspace / "package.json").exists():
            candidates = [
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ]
        else:
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
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return_code = proc.returncode
                    output = stdout.decode() + stderr.decode()
                    output += f"\n[exit code: {return_code}]"
                    return output
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    continue
            except FileNotFoundError:
                continue

        if (workspace / "pyproject.toml").exists():
            return "No test runner found. Consider installing pytest."
        if (workspace / "package.json").exists():
            return "No test runner found. Consider installing jest or vitest."
        return "No test runner found."
