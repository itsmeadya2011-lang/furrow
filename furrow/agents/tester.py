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
    from furrow.config import Settings

logger = logging.getLogger("furrow.tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        # First, run tests to see current state
        test_output = await self._run_tests()
        
        # Analyze test results with LLM
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        
        try:
            data = json.loads(response)
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])
        
        # If tests failed, try to fix them
        if not result.passed and result.failures:
            logger.info(f"Tests failed, attempting fix: {result.failures}")
            await self._attempt_fix(goal, tasks, result.failures, test_output)
            
            # Re-run tests after fix attempt
            test_output = await self._run_tests()
            prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output after fix attempt:\n{test_output}\n"
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
            
            try:
                data = json.loads(response)
                result = TestResult(**data)
            except (json.JSONDecodeError, ValueError):
                result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])
        
        return result

    async def _attempt_fix(self, goal: str, tasks: list[TaskModel], failures: list[str], test_output: str) -> None:
        """Attempt to fix failing tests by asking LLM to generate fixes."""
        # Get recent completed tasks for context
        completed_tasks = [t for t in tasks if t.status == "completed"]
        
        context_parts = []
        for t in completed_tasks[-3:]:  # Last 3 completed tasks
            context_parts.append(f"Task {t.id}: {t.description} -> {t.result[:200] if t.result else 'done'}")
        
        context = "\n".join(context_parts) if context_parts else "No recent tasks."
        
        fix_prompt = f"""You are a fixer agent. Tests are failing. Analyze the failures and provide fixes.

Goal: {goal}

Recent work:
{context}

Test failures:
{chr(10).join(failures)}

Full test output:
{test_output}

Return a JSON object with:
- "fixes": array of {{ "file": "path", "content": "fixed content" }}
- "summary": brief description of what was fixed
"""
        
        try:
            response = await self.client.complete(fix_prompt, model=self.client.settings.worker_model)
            data = json.loads(response)
            fixes = data.get("fixes", [])
            
            for fix in fixes:
                file_path = fix.get("file")
                content = fix.get("content")
                if file_path and content is not None:
                    await self.client.write_file(file_path, content)
                    logger.info(f"Applied fix to {file_path}")
        except Exception as e:
            logger.error(f"Fix attempt failed: {e}")

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
