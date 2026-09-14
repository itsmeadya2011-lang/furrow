from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

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

        # If tests failed, try to fix them
        if not result.passed and result.failures:
            result = await self._attempt_fix(goal, tasks, result, test_output)

        return result

    async def _attempt_fix(self, goal: str, tasks: list[TaskModel], result: TestResult, test_output: str) -> TestResult:
        console_msg = "[yellow]Tests failed. Attempting to fix...[/yellow]"
        print(console_msg)

        # Create fix tasks for each failure
        fix_tasks = []
        for i, failure in enumerate(result.failures[:3]):  # Limit to 3 fixes per cycle
            fix_task = TaskModel(
                id=f"fix_{i}",
                description=f"Fix test failure: {failure}",
                files=self._identify_files_from_failure(failure, tasks),
                dependencies=[],
            )
            fix_tasks.append(fix_task)

        # Run fix workers in parallel
        workers = [
            WorkerAgent(task=task, client=self.client, settings=self.settings).run()
            for task in fix_tasks
        ]
        fix_results = await asyncio.gather(*workers, return_exceptions=True)

        # Check if fixes worked by re-running tests
        for task, fix_result in zip(fix_tasks, fix_results):
            if isinstance(fix_result, Exception):
                print(f"[red]Fix task {task.id} failed: {fix_result}[/red]")
            else:
                print(f"[green]Fix task {task.id} completed: {fix_result[:100]}[/green]")

        # Re-run tests
        new_test_output = await self._run_tests()
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output after fix attempt:\n{new_test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _identify_files_from_failure(self, failure: str, tasks: list[TaskModel]) -> list[str]:
        """Extract file paths from test failure output."""
        files = []
        # Look for file paths in the failure
        import re
        matches = re.findall(r'([\w/.-]+\.py)', failure)
        for m in matches:
            if m not in files:
                files.append(m)
        # Also check task files
        for task in tasks:
            for f in task.files:
                if f not in files:
                    files.append(f)
        return files[:5]

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q", "--tb=short"],
            ["python", "-m", "pytest", "-q", "--tb=short"],
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