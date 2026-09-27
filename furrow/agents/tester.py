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
        task_descriptions = "\n".join(
            f"- Task {t.id}: {t.description}" for t in tasks
        )
        test_output = await self._run_tests()

        prompt = (
            f"Goal: {goal}\n\n"
            f"Tasks executed:\n{task_descriptions}\n\n"
            f"Test output:\n{test_output}\n"
        )
        response = await self.client.complete(
            prompt,
            system=TESTER_PROMPT,
            model=self.client.settings.tester_model,
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

    def _detect_test_command(self) -> list[str] | None:
        workspace = Path(self.client.settings.workspace)
        if (workspace / "pyproject.toml").exists():
            return ["pytest", "-q"]
        if (workspace / "setup.cfg").exists():
            return ["python", "-m", "pytest", "-q"]
        if (workspace / "setup.py").exists():
            return ["python", "-m", "pytest", "-q"]
        if (workspace / "pytest.ini").exists():
            return ["pytest", "-q"]
        if (workspace / "package.json").exists():
            if (workspace / "yarn.lock").exists():
                return ["yarn", "test", "--silent"]
            if (workspace / "pnpm-lock.yaml").exists():
                return ["pnpm", "test", "--silent"]
            return ["npm", "test", "--silent"]
        if (workspace / "Cargo.toml").exists():
            return ["cargo", "test"]
        if (workspace / "go.mod").exists():
            return ["go", "test", "./..."]
        return None

    async def _run_tests(self, timeout: float = 120) -> str:
        cmd = self._detect_test_command()
        if cmd is None:
            return "No test runner found."
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.client.settings.workspace,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                return (stdout.decode() if stdout else "") + (
                    stderr.decode() if stderr else ""
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Command timed out after {int(timeout)} seconds."
        except Exception as e:  # noqa: BLE001
            return f"Error running tests: {e}"
