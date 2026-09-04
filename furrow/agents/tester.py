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
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            lower = response.lower()
            negative = any(tok in lower for tok in ("fail", "error", "exception", "not ", "no "))
            passed = (
                "passed" in lower or "success" in lower or "all tests" in lower
            ) and not negative
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = getattr(self.client.settings, "workspace", None) or Path.cwd()
        candidates = self._candidates(workspace)
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
                    await proc.wait()
                    continue
            except (FileNotFoundError, PermissionError):
                continue
        return "No test runner found."

    def _candidates(self, workspace: Path) -> list[list[str]]:
        has_pytest = (
            (workspace / "pytest.ini").exists()
            or (workspace / "pyproject.toml").exists()
            or (workspace / "setup.cfg").exists()
        )
        has_js = (workspace / "package.json").exists()
        has_rust = (workspace / "Cargo.toml").exists()
        has_go = (workspace / "go.mod").exists()
        candidates: list[list[str]] = []
        if has_pytest:
            candidates += [["pytest", "-q"], ["python", "-m", "pytest", "-q"]]
        if has_js:
            candidates += [
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ]
        if has_rust:
            candidates += [["cargo", "test", "-q"]]
        if has_go:
            candidates += [["go", "test", "./..."]]
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
        return candidates
