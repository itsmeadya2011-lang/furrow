from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.logger = get_logger("tester")

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.logger.info("testing_start", goal=goal[:100], task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            self.logger.error("test_execution_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Include task context
        task_summary = "\n".join([f"  Task {t.id}: {t.description} - {t.status}" for t in tasks])
        
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nCompleted tasks:\n{task_summary}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            self.logger.info("testing_complete", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            self.logger.warning("test_parse_failed", response=response[:200], inferred_passed=passed)
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        """Run tests using the appropriate test runner for the project."""
        workspace = Path.cwd()
        
        # Detect project type and run appropriate tests
        if (workspace / "pyproject.toml").exists() or (workspace / "setup.py").exists() or (workspace / "requirements.txt").exists():
            return await self._run_python_tests()
        elif (workspace / "package.json").exists():
            return await self._run_node_tests()
        elif (workspace / "Cargo.toml").exists():
            return await self._run_rust_tests()
        elif (workspace / "go.mod").exists():
            return await self._run_go_tests()
        else:
            return await self._run_generic_tests()

    async def _run_python_tests(self) -> str:
        candidates = [
            ["pytest", "-q", "--tb=short"],
            ["python", "-m", "pytest", "-q", "--tb=short"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=Path.cwd()
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = f"Exit code: {proc.returncode}\n{stdout.decode()}\n{stderr.decode()}"
                    self.logger.debug("python_tests_run", cmd=cmd[0], exit_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No Python test runner found."

    async def _run_node_tests(self) -> str:
        candidates = [
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=Path.cwd()
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = f"Exit code: {proc.returncode}\n{stdout.decode()}\n{stderr.decode()}"
                    self.logger.debug("node_tests_run", cmd=cmd[0], exit_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No Node.js test runner found."

    async def _run_rust_tests(self) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "cargo", "test", "-q", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=Path.cwd()
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
            output = f"Exit code: {proc.returncode}\n{stdout.decode()}\n{stderr.decode()}"
            self.logger.debug("rust_tests_run", exit_code=proc.returncode)
            return output
        except (FileNotFoundError, Exception) as e:
            self.logger.error("rust_tests_failed", error=str(e))
            return f"Error running cargo test: {e}"

    async def _run_go_tests(self) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "go", "test", "./...", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=Path.cwd()
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
            output = f"Exit code: {proc.returncode}\n{stdout.decode()}\n{stderr.decode()}"
            self.logger.debug("go_tests_run", exit_code=proc.returncode)
            return output
        except (FileNotFoundError, Exception) as e:
            self.logger.error("go_tests_failed", error=str(e))
            return f"Error running go test: {e}"

    async def _run_generic_tests(self) -> str:
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
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=Path.cwd()
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = f"Exit code: {proc.returncode}\n{stdout.decode()}\n{stderr.decode()}"
                    self.logger.debug("generic_tests_run", cmd=cmd[0], exit_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
