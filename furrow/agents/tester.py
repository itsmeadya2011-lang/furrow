from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT, TESTER_FIX_PROMPT
from furrow.agents.worker import WorkerAgent
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

        # First, analyze test results
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])

        # If tests failed, try to fix them
        if not result.passed and result.failures:
            result = await self._fix_failures(goal, tasks, result, test_output)

        return result

    async def _fix_failures(self, goal: str, tasks: list[TaskModel], result: TestResult, test_output: str) -> TestResult:
        """Attempt to fix test failures by creating fix tasks and running workers."""
        self.client.console.print("[yellow]Attempting to fix test failures...[/yellow]") if hasattr(self.client, 'console') else None
        
        # Build context from failed tasks
        task_context = ""
        for task in tasks:
            if task.result:
                task_context += f"Task {task.id} ({task.description}):\n{task.result}\n\n"
        
        # Get relevant source files from tasks
        source_files = set()
        for task in tasks:
            for f in task.files:
                if not ("test" in f.lower() or f.startswith("tests/")):
                    source_files.add(f)
        
        # Read source files
        file_contents = {}
        for file_path in source_files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                pass
        
        files_context = ""
        if file_contents:
            files_context = "\n\nRelevant source files:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"
        
        # Ask LLM to generate fixes
        fix_prompt = f"{TESTER_FIX_PROMPT}\n\nGoal: {goal}\n\nTest failures:\n" + "\n".join(f"- {f}" for f in result.failures) + f"\n\nTest output:\n{test_output}\n{files_context}\n\nTask context:\n{task_context}\n\nReturn JSON with fixes:"
        
        response = await self.client.complete(fix_prompt, model=self.client.settings.tester_model)
        
        try:
            fix_data = json.loads(response)
            fixes = fix_data.get("fixes", [])
            
            # Apply fixes
            for fix in fixes:
                file_path = fix.get("file")
                content = fix.get("content", "")
                if file_path and content:
                    await self.client.write_file(file_path, content)
            
            # Re-run tests to verify fixes
            if fixes:
                test_output = await self._run_tests()
                prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output after fixes:\n{test_output}\n"
                response = await self.client.complete(prompt, model=self.client.settings.tester_model)
                try:
                    data = json.loads(response)
                    return TestResult(**data)
                except (json.JSONDecodeError, ValueError):
                    pass
            
            return result
        except (json.JSONDecodeError, ValueError):
            return result

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
