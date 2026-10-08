from __future__ import annotations

import aiofiles
import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from furrow.agents.prompts import TESTER_PROMPT, TESTER_FIX_PROMPT
from furrow.config import TaskModel, TestResult, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

console = Console()


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
            console.print("[yellow]Tests failed. Attempting to fix...[/yellow]")
            fixed = await self._fix_failures(goal, result.failures, test_output)
            if fixed:
                # Re-run tests after fix
                test_output = await self._run_tests()
                prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output after fix:\n{test_output}\n"
                response = await self.client.complete(prompt, model=self.client.settings.tester_model)
                try:
                    data = json.loads(response)
                    result = TestResult(**data)
                except (json.JSONDecodeError, ValueError):
                    result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])

        return result

    async def _fix_failures(self, goal: str, failures: list[str], test_output: str) -> bool:
        """Attempt to fix test failures by analyzing code and making changes."""
        # Get relevant file context
        file_context = await self._get_relevant_files()
        
        prompt = f"""{TESTER_FIX_PROMPT}

Goal: {goal}

Test failures:
{chr(10).join(f"- {f}" for f in failures)}

Full test output:
{test_output}

Relevant project files:
{file_context}

You must return a JSON object (not wrapped in markdown) with this shape:
{{
  "changes": [
    {{
      "file": "path/to/file.py",
      "content": "fixed file content",
      "operation": "create|update|delete"
    }}
  ],
  "summary": "Brief description of the fix applied",
  "confidence": 0.8
}}

If you cannot determine a fix, return an empty changes array with confidence 0.
"""

        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            confidence = data.get("confidence", 0)
            
            if changes and confidence > 0.5:
                for change in changes:
                    await self._apply_change(change)
                return True
        except (json.JSONDecodeError, ValueError):
            pass
        
        return False

    async def _get_relevant_files(self) -> str:
        """Get content of test files and recently modified files."""
        workspace = Path(self.settings.workspace)
        relevant = []
        
        # Look for test files
        for pattern in ["**/test_*.py", "**/*_test.py", "**/tests/**/*.py"]:
            for f in workspace.glob(pattern):
                try:
                    content = await self.client.read_file(f)
                    relevant.append(f"=== {f.relative_to(workspace)} ===\n{content}")
                except Exception:
                    pass
        
        return "\n\n".join(relevant[:10]) if relevant else "No test files found."

    async def _apply_change(self, change: dict) -> None:
        """Apply a file change."""
        file_path = Path(self.settings.workspace) / change["file"]
        operation = change.get("operation", "update")
        content = change.get("content", "")

        if operation == "delete":
            if file_path.exists():
                file_path.unlink()
        elif operation == "create":
            file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(file_path, "w") as f:
                await f.write(content)
        elif operation == "update":
            file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(file_path, "w") as f:
                await f.write(content)

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