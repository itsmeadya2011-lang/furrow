from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("testing_started", goal=goal[:100])
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_runner_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        try:
            data = json.loads(response)
            passed = data.get("passed", False)
            summary = data.get("summary", "")
            failures = data.get("failures", [])
            fixes = data.get("fixes", [])

            # If tests failed and we have fixes, try to apply them
            if not passed and fixes:
                logger.info("applying_fixes", fix_count=len(fixes))
                await self._apply_fixes(fixes, goal)

            logger.info("testing_completed", passed=passed, failures=len(failures))
            return TestResult(passed=passed, summary=summary, failures=failures)
        except (json.JSONDecodeError, ValueError):
            logger.warning("tester_json_parse_failed")
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _apply_fixes(self, fixes: list[dict], goal: str) -> None:
        """Apply fixes by creating worker tasks for each fix."""
        for fix in fixes:
            description = fix.get("description", "")
            files = fix.get("files", [])
            if not description:
                continue
            task = TaskModel(
                id=f"fix-{len(fixes)}",
                description=description,
                files=files,
            )
            worker = WorkerAgent(task=task, client=self.client)
            try:
                await worker.run()
            except Exception as e:
                logger.error("fix_application_failed", error=str(e))

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
