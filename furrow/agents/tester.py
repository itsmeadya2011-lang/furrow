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
        try:
            test_output, exit_code = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if exit_code != 0:
            lines = test_output.splitlines()
            tail = "\n".join(lines[-50:]) if lines else test_output
            return TestResult(
                passed=False,
                summary=f"Tests failed (exit {exit_code})",
                failures=[tail] if tail else [],
            )

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> tuple[str, int]:
        candidates = []
        if os.path.exists("package.json"):
            candidates.extend(
                [
                    ["npm", "test", "--", "--silent"],
                    ["pnpm", "test", "--", "--silent"],
                    ["yarn", "test", "--silent"],
                ]
            )
        if os.path.exists("pyproject.toml") or os.path.exists("setup.py") or os.path.exists("setup.cfg"):
            candidates.extend(
                [
                    ["pytest", "-q"],
                    ["python", "-m", "pytest", "-q"],
                ]
            )
        if os.path.exists("Cargo.toml"):
            candidates.append(["cargo", "test", "-q"])
        if os.path.exists("go.mod"):
            candidates.append(["go", "test", "./..."])

        if not candidates:
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]

        last_output = ""
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                except asyncio.TimeoutError:
                    proc.kill()
                    return "Test runner timed out.", 124
            except (FileNotFoundError, Exception):
                continue
            last_output = stdout.decode() + stderr.decode()
            return_code = proc.returncode if proc.returncode is not None else 0
            return last_output, return_code
        return last_output, 0
