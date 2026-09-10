from __future__ import annotations

import asyncio
import json
import os
import sys
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

        # Include task context in the prompt
        task_context = ""
        if tasks:
            task_context = "\n\nTasks in this cycle:\n"
            for task in tasks:
                task_context += f"  - {task.id}: {task.description} (status: {task.status})\n"
                if task.result:
                    task_context += f"    Result: {task.result[:200]}...\n"

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}{task_context}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        # Detect project type and run appropriate tests
        project_type = self._detect_project_type()
        
        candidates = self._get_test_commands(project_type)
        
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
                    elif proc.returncode == 0:
                        return "Tests passed (no output)"
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return f"No test runner found for {project_type} project."

    def _detect_project_type(self) -> str:
        """Detect the project type based on files in the workspace."""
        cwd = Path.cwd()
        if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists() or (cwd / "requirements.txt").exists():
            return "python"
        elif (cwd / "package.json").exists():
            return "javascript"
        elif (cwd / "Cargo.toml").exists():
            return "rust"
        elif (cwd / "go.mod").exists():
            return "go"
        elif (cwd / "pom.xml").exists() or (cwd / "build.gradle").exists():
            return "java"
        return "unknown"

    def _get_test_commands(self, project_type: str) -> list[list[str]]:
        """Get test commands for the detected project type."""
        commands = {
            "python": [
                ["python", "-m", "pytest", "-v", "--tb=short"],
                ["pytest", "-v", "--tb=short"],
                ["python", "-m", "unittest", "discover", "-v"],
            ],
            "javascript": [
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ],
            "rust": [
                ["cargo", "test", "--", "--nocapture"],
            ],
            "go": [
                ["go", "test", "./...", "-v"],
            ],
            "java": [
                ["./mvnw", "test", "-q"],
                ["mvn", "test", "-q"],
                ["./gradlew", "test", "--quiet"],
                ["gradle", "test", "--quiet"],
            ],
        }
        return commands.get(project_type, [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ])
