from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class TesterAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | str | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings
        self.workspace = Path(workspace) if workspace else (settings.workspace if settings else None)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        workspace_info = ""
        if self.workspace:
            workspace_info = f"\nWorkspace: {self.workspace}\n"

        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("Test execution failed: %s", e)
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if not test_output or test_output.strip() == "No test runner found.":
            logger.warning("No test runner found. Running goal-specific validation prompt.")
            prompt = (
                f"{TESTER_PROMPT}{workspace_info}\n\n"
                f"Goal: {goal}\n\n"
                f"No automated tests were found or run. "
                f"Based on the task descriptions and the workspace structure, "
                f"determine if the work appears complete and correct.\n\n"
                f"Tasks executed:\n"
                + "\n".join(f"- {t.id}: {t.description} ({t.status})" for t in tasks)
                + "\n"
            )
        else:
            prompt = (
                f"{TESTER_PROMPT}{workspace_info}\n\n"
                f"Goal: {goal}\n\n"
                f"Test output:\n{test_output}\n"
            )

        try:
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        except Exception as e:
            logger.error("Tester LLM call failed: %s", e)
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["python", "-m", "unittest", "discover", "-v"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
            ["dotnet", "test", "--no-build"],
            ["mvn", "test", "-q"],
            ["gradle", "test", "--quiet"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.workspace) if self.workspace else None,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    logger.warning("Test command %s timed out", cmd)
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
