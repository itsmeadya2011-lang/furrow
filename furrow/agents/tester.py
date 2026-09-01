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

# Manifest files that indicate a project type and their corresponding test runners
PROJECT_MANIFESTS: dict[str, list[list[str]]] = {
    "pyproject.toml": [["pytest", "-q"], ["python", "-m", "pytest", "-q"]],
    "setup.py": [["pytest", "-q"], ["python", "-m", "pytest", "-q"]],
    "setup.cfg": [["pytest", "-q"], ["python", "-m", "pytest", "-q"]],
    "package.json": [["npm", "test", "--", "--silent"], ["pnpm", "test", "--", "--silent"], ["yarn", "test", "--silent"]],
    "Cargo.toml": [["cargo", "test", "-q"]],
    "go.mod": [["go", "test", "./..."]],
}


def detect_test_commands(workspace: Path | None = None) -> list[list[str]]:
    """Detect test commands based on project manifest files."""
    if workspace is None:
        workspace = Path.cwd()

    # Check for manifest files in priority order
    for manifest, commands in PROJECT_MANIFESTS.items():
        if (workspace / manifest).exists():
            return commands

    # Fallback: try all commands
    return [
        ["pytest", "-q"],
        ["python", "-m", "pytest", "-q"],
        ["npm", "test", "--", "--silent"],
        ["pnpm", "test", "--", "--silent"],
        ["yarn", "test", "--silent"],
        ["cargo", "test", "-q"],
        ["go", "test", "./..."],
    ]


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
        workspace = self.client.settings.workspace
        candidates = detect_test_commands(workspace)
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace),
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
