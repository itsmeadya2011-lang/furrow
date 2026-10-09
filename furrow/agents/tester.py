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
        project_type = self._detect_project_type(workspace)

        if project_type == "python":
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["python", "-m", "unittest", "discover", "-q"],
            ]
        elif project_type == "node":
            candidates = [
                ["npm", "test", "--", "--silent"],
                ["npx", "jest", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ]
        elif project_type == "rust":
            candidates = [
                ["cargo", "test", "-q"],
            ]
        elif project_type == "go":
            candidates = [
                ["go", "test", "./..."],
            ]
        else:
            # Unknown project type - try common runners
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
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
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode == 0 or "no tests" in output.lower():
                        return output
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    continue
            except (FileNotFoundError, Exception):
                continue

        return "No test runner found."

    def _detect_project_type(self, workspace: Path) -> str:
        """Detect the project type based on files in the workspace."""
        if not workspace.exists():
            return "unknown"

        files = [f.name for f in workspace.iterdir() if f.is_file()]

        # Python indicators
        python_indicators = {
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "Pipfile",
            "poetry.lock",
        }
        if files & python_indicators:
            return "python"

        # Node.js indicators
        node_indicators = {"package.json", "yarn.lock", "package-lock.json", "pnpm-lock.yaml"}
        if files & node_indicators:
            return "node"

        # Rust indicators
        if "Cargo.toml" in files:
            return "rust"

        # Go indicators
        if "go.mod" in files:
            return "go"

        return "unknown"