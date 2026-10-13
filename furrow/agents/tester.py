from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_FIX_PROMPT, TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
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

    async def fix_failures(self, goal: str, tasks: list[TaskModel], test_output: str, modified_files: list[str]) -> dict:
        """Attempt to fix test failures by analyzing and modifying files."""
        # Read modified files for context
        file_contents = {}
        for file_path in modified_files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = f"[Could not read {file_path}]"

        context = ""
        if file_contents:
            context += "\nModified files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{TESTER_FIX_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            return json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return {"fixes": [], "summary": "Failed to parse fix response"}

    async def _run_tests(self) -> str:
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
