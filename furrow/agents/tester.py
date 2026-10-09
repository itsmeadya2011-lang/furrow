from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from rich.console import Console

from furrow.agents.prompts import TESTER_PROMPT
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

console = Console()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings

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
        if not result.passed and result.failures:
            result = await self._fix_failures(goal, result, tasks)

        return result

    async def _fix_failures(self, goal: str, result: TestResult, tasks: list[TaskModel]) -> TestResult:
        """Attempt to fix failing tests by creating fix tasks."""
        console_output = []
        
        for failure in result.failures:
            console_output.append(f"Attempting to fix: {failure}")
            
            # Create a fix task
            fix_task = TaskModel(
                id=f"fix_{len(tasks) + 1}",
                description=f"Fix test failure: {failure}",
                files=[],  # Will be determined by worker
                dependencies=[],
            )
            
            # Run worker to fix
            worker = WorkerAgent(task=fix_task, client=self.client, settings=self.settings)
            fix_result = await worker.run()
            console_output.append(f"Fix attempt result: {fix_result}")
        
        # Re-run tests after fixes
        test_output = await self._run_tests()
        
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output after fixes:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            new_result = TestResult(**data)
            new_result.summary = "\n".join(console_output) + "\n\n" + new_result.summary
            return new_result
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower(), 
                summary="\n".join(console_output) + "\n\n" + response, 
                failures=[]
            )

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
