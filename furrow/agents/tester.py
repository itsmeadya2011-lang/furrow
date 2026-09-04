from __future__ import annotations

import asyncio
import json
import os
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
        except asyncio.TimeoutError:
            return TestResult(passed=False, summary="Test runner timed out", failures=["Timeout"])
        except FileNotFoundError:
            return TestResult(passed=False, summary="No test runner found", failures=["No test runner found"])
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if not test_output.strip():
            return TestResult(passed=True, summary="No tests or no output from test runner", failures=[])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = Path(self.client.settings.workspace)
        candidates = self._detect_test_commands(workspace)
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
            except FileNotFoundError:
                continue
        return "No test runner found."

    def _detect_test_commands(self, workspace: Path) -> list[list[str]]:
        commands: list[list[str]] = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
        ]
        if (workspace / "package.json").exists():
            commands.extend([
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ])
        if (workspace / "Cargo.toml").exists():
            commands.append(["cargo", "test", "-q"])
        if (workspace / "go.mod").exists():
            commands.append(["go", "test", "./..."])
        return commands
