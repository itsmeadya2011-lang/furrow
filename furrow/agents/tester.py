from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger()

TEST_TIMEOUT = 120


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        workspace = self.client.settings.workspace
        test_output = ""
        try:
            test_output = await self._run_tests(workspace)
        except Exception as e:  # noqa: BLE001
            log.error("test execution failed", error=str(e), goal=goal)
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        task_lines = [
            f"- [{t.id}] {t.description}" for t in tasks
        ]
        task_context = "\n".join(task_lines) if task_lines else "none"

        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Tasks implemented:\n{task_context}\n\n"
            f"Workspace: {workspace}\n\n"
            f"Test output:\n{test_output}\n"
        )
        response = await self.client.complete(
            prompt, model=self.client.settings.tester_model
        )
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower(),
                summary=response,
                failures=[],
            )

    @staticmethod
    def _detect_test_command(workspace: Path) -> list[str] | None:
        if (workspace / "pyproject.toml").exists():
            if shutil.which("pytest"):
                return ["pytest", "-q"]
            if shutil.which("python"):
                return ["python", "-m", "pytest", "-q"]
        if (workspace / "package.json").exists():
            if (workspace / "pnpm-lock.yaml").exists() and shutil.which("pnpm"):
                return ["pnpm", "test", "--", "--silent"]
            if (workspace / "yarn.lock").exists() and shutil.which("yarn"):
                return ["yarn", "test", "--silent"]
            if shutil.which("npm"):
                return ["npm", "test", "--", "--silent"]
        if (workspace / "Cargo.toml").exists() and shutil.which("cargo"):
            return ["cargo", "test", "-q"]
        if (workspace / "go.mod").exists() and shutil.which("go"):
            return ["go", "test", "./..."]
        if (workspace / "Makefile").exists() and shutil.which("make"):
            return ["make", "test"]
        return None

    async def _run_tests(self, workspace: Path) -> str:
        command = self._detect_test_command(workspace)
        if command is None:
            log.warning("no test runner detected", workspace=str(workspace))
            return "No test runner found."

        log.info("running tests", command=" ".join(command), workspace=str(workspace))
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return f"Command not found: {command[0]}"

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=TEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log.warning("test command timed out", command=" ".join(command))
            return f"Test command timed out after {TEST_TIMEOUT}s: {command[0]}"

        return stdout.decode() + stderr.decode()
