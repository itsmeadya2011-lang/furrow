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

    def _detect_project_type(self) -> str:
        cwd = os.getcwd()
        # Check JavaScript first so a project with both package.json and
        # pyproject.toml (e.g. monorepos) doesn't get classified as Python.
        if os.path.exists(os.path.join(cwd, "package.json")):
            if os.path.exists(os.path.join(cwd, "pnpm-lock.yaml")):
                return "javascript-pnpm"
            if os.path.exists(os.path.join(cwd, "yarn.lock")):
                return "javascript-yarn"
            return "javascript-npm"
        for marker in ("pyproject.toml", "pytest.ini", "setup.py"):
            if os.path.exists(os.path.join(cwd, marker)):
                return "python"
        if os.path.exists(os.path.join(cwd, "Cargo.toml")):
            return "rust"
        if os.path.exists(os.path.join(cwd, "go.mod")):
            return "go"
        return "unknown"

    async def _run_tests(self) -> str:
        project_type = self._detect_project_type()
        if project_type == "python":
            candidates: list[list[str]] = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
            ]
        elif project_type == "javascript-npm":
            candidates = [["npm", "test", "--", "--silent"]]
        elif project_type == "javascript-pnpm":
            candidates = [["pnpm", "test", "--", "--silent"]]
        elif project_type == "javascript-yarn":
            candidates = [["yarn", "test", "--silent"]]
        elif project_type == "rust":
            candidates = [["cargo", "test", "-q"]]
        elif project_type == "go":
            candidates = [["go", "test", "./..."]]
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
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
