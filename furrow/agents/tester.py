from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import aiofiles

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class TesterAgent:
    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self._timeout: int = 120

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(
                passed=False, summary=str(e), failures=[str(e)]
            )

        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
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

    async def _run_tests(self) -> str:
        candidates = self._detect_test_commands()
        if not candidates:
            return "No test runner found."

        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                # Command not on PATH — try next candidate
                continue
            except Exception as e:
                logger.warning("Failed to start %s: %s", " ".join(cmd), e)
                continue

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout
                )
                output = (stdout.decode(errors="replace")
                          + stderr.decode(errors="replace"))
                return_code = proc.returncode or 0
                header = f"Command: {' '.join(cmd)} (exit code: {return_code})\n"
                return header + output
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("Test command timed out after %ds: %s", self._timeout, " ".join(cmd))
                return f"Test command timed out after {self._timeout}s: {' '.join(cmd)}"

        return "No test runner found."

    def _detect_test_commands(self) -> list[list[str]]:
        """Detect available test runners based on project files."""
        import shutil

        commands: list[list[str]] = []

        # Python projects
        if shutil.which("pytest"):
            commands.append(["pytest", "-q"])
        elif shutil.which("python"):
            # Verify pytest module is available
            import subprocess as sync_subprocess

            check = sync_subprocess.run(
                ["python", "-m", "pytest", "--version"],
                capture_output=True,
            )
            if check.returncode == 0:
                commands.append(["python", "-m", "pytest", "-q"])

        # Node.js projects
        for runner in ["npm", "pnpm", "yarn"]:
            if shutil.which(runner):
                commands.append([runner, "test", "--silent"])

        # Rust projects
        if shutil.which("cargo"):
            commands.append(["cargo", "test", "-q"])

        # Go projects
        if shutil.which("go"):
            commands.append(["go", "test", "./..."])

        return commands
