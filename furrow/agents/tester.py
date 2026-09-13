from __future__ import annotations

import asyncio
import json
import shlex
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
        self.settings = self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        project_type = self._detect_project_type()
        sections = [
            ("Project type", project_type),
            ("Tests", await self._run_tests(project_type)),
            ("Lint", await self._run_lint(project_type)),
            ("Type check", await self._run_types(project_type)),
        ]
        test_output = "\n\n".join(f"=== {title.upper()} ===\n{output}" for title, output in sections)

        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n"
            f"=== TEST OUTPUT ===\n{test_output}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _detect_project_type(self) -> str:
        workspace = self.settings.workspace
        if self._has_any(workspace, ["pyproject.toml", "setup.py", "setup.cfg"]):
            return "python"
        if (workspace / "package.json").exists():
            return "node"
        if (workspace / "Cargo.toml").exists():
            return "rust"
        if (workspace / "go.mod").exists():
            return "go"
        return "unknown"

    async def _run_tests(self, project_type: str) -> str:
        candidates_by_type = {
            "python": [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
            ],
            "node": self._node_test_commands(),
            "rust": [["cargo", "test", "-q"]],
            "go": [["go", "test", "./..."]],
            "unknown": [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ],
        }
        return await self._run_first_command(candidates_by_type[project_type], "No test runner found.")

    async def _run_lint(self, project_type: str) -> str:
        candidates_by_type = {
            "python": [
                ["ruff", "check"],
                ["python", "-m", "ruff", "check"],
            ],
            "node": self._node_lint_commands(),
            "rust": [["cargo", "clippy"]],
            "go": [["go", "vet"]],
            "unknown": [],
        }
        return await self._run_first_command(candidates_by_type[project_type], "")

    async def _run_types(self, project_type: str) -> str:
        candidates_by_type = {
            "python": [
                ["mypy"],
                ["python", "-m", "mypy"],
            ],
            "node": [["npm", "run", "typecheck"], ["tsc", "--noEmit"]],
            "rust": [["cargo", "check"]],
            "go": [],
            "unknown": [],
        }
        if project_type == "node":
            return await self._run_node_typecheck()
        return await self._run_first_command(candidates_by_type[project_type], "")

    async def _run_node_typecheck(self) -> str:
        output = await self._run_first_command([["npm", "run", "typecheck"]], "")
        if output and not self._should_retry_node_script(output):
            return output
        return await self._run_first_command([["tsc", "--noEmit"]], "")

    async def _run_first_command(
        self,
        candidates: list[list[str]],
        not_found_message: str,
    ) -> str:
        for cmd in candidates:
            output = await self._run_command(cmd)
            if output is None:
                continue
            return f"Command: {shlex.join(cmd)}\n{output}"
        return not_found_message

    async def _run_command(self, cmd: list[str]) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except (FileNotFoundError, OSError):
            return None
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            return stdout.decode() + stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "Timed out after 120 seconds."

    def _has_any(self, workspace: Path, filenames: list[str]) -> bool:
        return any((workspace / filename).exists() for filename in filenames)

    def _node_package_manager(self) -> str:
        workspace = self.settings.workspace
        if (workspace / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (workspace / "yarn.lock").exists():
            return "yarn"
        return "npm"

    def _node_test_commands(self) -> list[list[str]]:
        manager = self._node_package_manager()
        if manager == "pnpm":
            return [["pnpm", "test", "--", "--silent"], ["npm", "test", "--", "--silent"]]
        if manager == "yarn":
            return [["yarn", "test", "--silent"], ["npm", "test", "--", "--silent"]]
        return [["npm", "test", "--", "--silent"]]

    def _node_lint_commands(self) -> list[list[str]]:
        manager = self._node_package_manager()
        if manager == "pnpm":
            return [["pnpm", "run", "lint"], ["npm", "run", "lint"]]
        if manager == "yarn":
            return [["yarn", "run", "lint"], ["npm", "run", "lint"]]
        return [["npm", "run", "lint"]]

    def _should_retry_node_script(self, output: str) -> bool:
        lowered = output.lower()
        return "missing script" in lowered or "unknown command" in lowered
