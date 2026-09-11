from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_FIX_PROMPT, TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("furrow.tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("testing_started", task_count=len(tasks))
        test_output = await self._run_tests()

        # First, evaluate test results
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        try:
            data = json.loads(response)
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = TestResult(
                passed="passed" in response.lower() and "failed" not in response.lower(),
                summary=response,
                failures=[],
            )

        logger.info("test_evaluation", passed=result.passed, summary=result.summary, failure_count=len(result.failures))

        # If tests failed, attempt to fix them
        if not result.passed and result.failures:
            await self._fix_failures(goal, test_output, tasks)

        return result

    async def _fix_failures(self, goal: str, test_output: str, tasks: list[TaskModel]) -> None:
        """Attempt to fix failing tests by modifying source files."""
        logger.info("fix_attempt_started", failure_count=len(tasks))
        # Collect relevant files from tasks
        relevant_files = set()
        for task in tasks:
            relevant_files.update(task.files)

        # Read file contents
        file_contents = {}
        for file_path in relevant_files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""
            except Exception as e:
                file_contents[file_path] = f"Error reading: {e}"

        files_context = "\n\n".join(
            f"=== {path} ===\n{content}" for path, content in file_contents.items()
        )

        prompt = f"{TESTER_FIX_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n\nRelevant files:\n{files_context}\n"

        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        try:
            data = json.loads(response)
            files_to_write = data.get("files", [])
            for file_data in files_to_write:
                path = file_data.get("path")
                content = file_data.get("content", "")
                if path:
                    await self.client.write_file(path, content)
                    logger.debug("fix_file_written", path=path, size=len(content))
            logger.info("fix_attempt_completed", files_fixed=len(files_to_write))
        except (json.JSONDecodeError, ValueError):
            logger.warning("fix_attempt_failed", response=response[:200])
            # Fix attempt failed, will retry in next cycle
            pass

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
                    output = stdout.decode() + stderr.decode()
                    logger.debug("test_run_completed", command=cmd[0], output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", command=cmd[0])
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_runner_not_found", command=cmd[0], error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."