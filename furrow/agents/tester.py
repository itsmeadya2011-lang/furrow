from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - intentional: any test runner error should fail gracefully
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(extract_json(response))
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _test_candidates(self) -> list[list[str]]:
        root = Path(self.client.settings.workspace)
        python_markers = ("pyproject.toml", "requirements.txt", "setup.py")
        if any((root / marker).exists() for marker in python_markers) or (root / "tests").is_dir():
            return self._pytest_candidates()
        if (root / "package.json").exists():
            return [
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ]
        if (root / "Cargo.toml").exists():
            return [["cargo", "test", "-q"]]
        if (root / "go.mod").exists():
            return [["go", "test", "./..."]]
        return self._pytest_candidates()

    @staticmethod
    def _pytest_candidates() -> list[list[str]]:
        return [
            [sys.executable, "-m", "pytest", "-q"],
            ["python3", "-m", "pytest", "-q"],
            ["pytest", "-q"],
        ]

    async def _run_tests(self) -> str:
        for cmd in self._test_candidates():
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.client.settings.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    continue
            except (FileNotFoundError, OSError):
                continue
        return "No test runner found."
