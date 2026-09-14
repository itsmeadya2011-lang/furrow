from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("tester_started", goal=goal, num_tasks=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Provide context about what was changed
        task_context = ""
        if tasks:
            task_context = "\n\nTasks executed this cycle:\n"
            for task in tasks:
                task_context += f"  - {task.id}: {task.description} ({task.status})\n"

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}{task_context}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester_completed", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            logger.warning("tester_parse_failed", response=response[:500])
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = self.settings.workspace if self.settings else Path.cwd()
        
        # Auto-detect project type and run appropriate tests
        test_commands = self._detect_test_commands(workspace)
        logger.info("test_commands_detected", commands=[str(c) for c in test_commands])
        
        for cmd in test_commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, 
                    stdout=asyncio.subprocess.PIPE, 
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode == 0 or output.strip():
                        logger.info("test_command_executed", cmd=str(cmd), returncode=proc.returncode)
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_command_timeout", cmd=str(cmd))
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_command_not_found", cmd=str(cmd), error=str(e))
                continue
        return "No test runner found or tests not configured."

    def _detect_test_commands(self, workspace: Path) -> list[list[str]]:
        commands = []
        
        # Python
        if (workspace / "pyproject.toml").exists() or (workspace / "setup.py").exists() or (workspace / "requirements.txt").exists():
            commands.extend([
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
            ])
        
        # JavaScript/TypeScript
        if (workspace / "package.json").exists():
            commands.extend([
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ])
        
        # Rust
        if (workspace / "Cargo.toml").exists():
            commands.append(["cargo", "test", "-q"])
        
        # Go
        if (workspace / "go.mod").exists():
            commands.append(["go", "test", "./..."])
        
        # Default fallbacks
        commands.extend([
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
        ])
        
        return commands
