from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import FIXER_PROMPT, TESTER_PROMPT
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
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])

        # If tests failed, attempt to fix them
        if not result.passed:
            await self._fix_failures(goal, tasks, test_output)
            # Re-run tests to verify fix
            test_output = await self._run_tests()
            prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
            try:
                data = json.loads(response)
                result = TestResult(**data)
            except (json.JSONDecodeError, ValueError):
                result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])

        return result

    async def _fix_failures(self, goal: str, tasks: list[TaskModel], test_output: str) -> None:
        # Collect files from tasks to fix
        files_to_fix = set()
        for task in tasks:
            files_to_fix.update(task.files)
        
        # Read those files
        file_contents = {}
        for file_path in files_to_fix:
            path = Path(file_path)
            if path.exists():
                file_contents[file_path] = await self.client.read_file(path)

        files_info = "\n\n".join(
            f"=== {path} ===\n{content}"
            for path, content in file_contents.items()
        )

        prompt = f"{FIXER_PROMPT}\n\nGoal: {goal}\n\nTest failures:\n{test_output}\n\nFiles:\n{files_info}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            
            # Apply fixes
            for change in changes:
                path = Path(change["path"])
                content = change["content"]
                await self.client.write_file(path, content)
        except (json.JSONDecodeError, KeyError, ValueError):
            # If fixer fails, continue without fixing
            pass

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
