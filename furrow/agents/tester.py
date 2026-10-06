from __future__ import annotations

import asyncio
import json
import os
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
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # If tests passed, return success
        if "passed" in test_output.lower() and "failed" not in test_output.lower():
            return TestResult(passed=True, summary="All tests passed", failures=[])

        # Tests failed - try to fix them
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n\nReturn JSON with fixes to apply:\n{{\n  \"fixes\": [\n    {{\n      \"file\": \"path/to/file.py\",\n      \"content\": \"fixed file content\"\n    }}\n  ],\n  \"summary\": \"Description of fixes applied\"\n}}"
        
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        
        try:
            data = json.loads(response)
            fixes = data.get("fixes", [])
            summary = data.get("summary", "Applied fixes for test failures")
            
            # Apply fixes
            for fix in fixes:
                file_path = fix.get("file")
                content = fix.get("content")
                if file_path and content is not None:
                    await self.client.write_file(file_path, content)
            
            # Re-run tests after fixes
            test_output = await self._run_tests()
            passed = "passed" in test_output.lower() and "failed" not in test_output.lower()
            
            if passed:
                return TestResult(passed=True, summary=f"Fixed and tests now pass: {summary}", failures=[])
            else:
                return TestResult(passed=False, summary=f"Fixes applied but tests still fail: {summary}", failures=[test_output])
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed=False, summary="Failed to parse tester response", failures=[test_output])

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
