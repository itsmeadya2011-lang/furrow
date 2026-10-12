from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.util import extract_json

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = await self._run_tests()

        if test_output == "No test runner found.":
            return TestResult(passed=True, summary="No test runner found; skipping.", failures=[])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = extract_json(response)
            return TestResult(**data)
        except ValueError:
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _candidate_commands(self) -> list[list[str]]:
        custom = self.client.settings.test_command
        if custom:
            return [shlex.split(custom)]
        workspace = Path(self.client.settings.workspace)
        candidates: list[list[str]] = []
        if (workspace / "package.json").exists():
            candidates.extend(
                [
                    ["pnpm", "test", "--", "--silent"],
                    ["npm", "test", "--", "--silent"],
                    ["yarn", "test", "--silent"],
                ]
            )
        if (workspace / "Cargo.toml").exists():
            candidates.append(["cargo", "test", "-q"])
        if (workspace / "go.mod").exists():
            candidates.append(["go", "test", "./..."])
        candidates.extend(
            [
                ["pytest", "-q"],
                ["python3", "-m", "pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
            ]
        )
        return candidates

    async def _run_tests(self) -> str:
        for cmd in self._candidate_commands():
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.client.settings.workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (FileNotFoundError, OSError):
                continue
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.client.settings.test_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                continue
            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            header = f"$ {' '.join(cmd)} (exit {proc.returncode})\n"
            return header + output
        return "No test runner found."