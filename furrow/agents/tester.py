from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult, Settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("Running tests", task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests(tasks)
        except Exception as e:
            logger.error("Test execution failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("Test evaluation complete", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse test result, using fallback")
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self, tasks: list[TaskModel]) -> str:
        """Run tests and return output. Tries to run tests relevant to changed files."""
        # Get test files from tasks
        test_files = []
        for task in tasks:
            for f in task.files:
                if f.startswith("test") or f.endswith("_test.py") or f.startswith("tests/"):
                    test_files.append(f)
        
        # Build test commands
        candidates = self._get_test_commands(test_files)
        
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found or no tests to run."

    def _get_test_commands(self, test_files: list[str]) -> list[list[str]]:
        """Generate test commands based on project type and test files."""
        candidates = []
        cwd = Path.cwd()
        
        # Check for Python project
        if (cwd / "pyproject.toml").exists() or (cwd / "requirements.txt").exists() or (cwd / "setup.py").exists():
            if test_files:
                for tf in test_files:
                    candidates.append(["python", "-m", "pytest", tf, "-v"])
            candidates.extend([
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
            ])
        
        # Check for Node.js project
        if (cwd / "package.json").exists():
            if test_files:
                for tf in test_files:
                    candidates.append(["npm", "test", "--", tf, "--silent"])
            candidates.extend([
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ])
        
        # Check for Rust project
        if (cwd / "Cargo.toml").exists():
            candidates.append(["cargo", "test", "-q"])
        
        # Check for Go project
        if (cwd / "go.mod").exists():
            candidates.append(["go", "test", "./..."])
        
        return candidates
