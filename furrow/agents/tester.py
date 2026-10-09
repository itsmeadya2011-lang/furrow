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


def detect_test_commands(root: Path | None = None) -> list[list[str]]:
    if root is None:
        root = Path.cwd()

    commands: list[list[str]] = []

    pytest_candidates = [
        ["pytest", "-q"],
        ["python", "-m", "pytest", "-q"],
    ]
    has_pytest_project = any(
        (root / name).exists()
        for name in ("pyproject.toml", "pytest.ini", "tox.ini")
    ) or (root / "tests").is_dir()
    if has_pytest_project:
        commands.extend(pytest_candidates)

    if (root / "package.json").exists():
        commands.extend(
            [
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ]
        )

    if (root / "Cargo.toml").exists():
        commands.append(["cargo", "test", "-q"])

    if (root / "go.mod").exists():
        commands.append(["go", "test", "./..."])

    if not any(cmd[0] == "pytest" for cmd in commands):
        commands.extend(pytest_candidates)

    return commands


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
        root = self.client.settings.workspace if self.client is not None else None
        for cmd in detect_test_commands(root):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(root) if root is not None else None,
                )
            except FileNotFoundError:
                continue

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                return stdout.decode(errors="replace") + stderr.decode(errors="replace")
            except asyncio.TimeoutError:
                proc.kill()
                continue

        return "No test runner found."