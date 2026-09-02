from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT, TESTER_USER_TEMPLATE
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

        task_summary = "\n".join(
            f"- {t.id}: {t.description} [{t.status}]" for t in tasks
        ) or "(none)"
        prompt = TESTER_USER_TEMPLATE.format(goal=goal, tasks=task_summary, test_output=test_output)
        response = await self.client.complete(
            prompt, system=TESTER_PROMPT, model=self.client.settings.tester_model
        )
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        if (
            Path("pyproject.toml").exists()
            or Path("pytest.ini").exists()
            or Path("setup.py").exists()
        ):
            cmd = ["pytest", "-q"]
        elif Path("package.json").exists():
            cmd = ["npm", "test", "--", "--silent"]
        elif Path("Cargo.toml").exists():
            cmd = ["cargo", "test", "-q"]
        elif Path("go.mod").exists():
            cmd = ["go", "test", "./..."]
        else:
            return "No test runner found."
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except (FileNotFoundError, OSError):
            return "No test runner found."
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            return "Tests timed out after 120s."
        output = stdout.decode() + stderr.decode()
        if len(output) > 4000:
            output = output[-4000:]
        return output