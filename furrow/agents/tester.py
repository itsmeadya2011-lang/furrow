from __future__ import annotations

import asyncio
import json
import shlex
from typing import TYPE_CHECKING

from furrow.agents.jsonparse import extract_json
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def run(self, goal: str, tasks: list[TaskModel], context: str = "") -> TestResult:
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n"
        if context:
            prompt += f"\nWork already done:\n{context}\n"
        prompt += f"\nTest output:\n{test_output}\n"

        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(extract_json(response))
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower(), summary=response.strip(), failures=[]
            )

    async def _run_tests(self) -> str:
        if self.settings.test_command:
            candidates = [shlex.split(self.settings.test_command)]
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
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.settings.workspace,
                )
            except (FileNotFoundError, OSError):
                continue
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.settings.test_timeout
                )
                return stdout.decode(errors="replace") + stderr.decode(errors="replace")
            except (asyncio.TimeoutError, TimeoutError):
                proc.kill()
                await proc.wait()
                continue
        return "No test runner found."
