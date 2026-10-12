from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings as SettingsType

logger = logging.getLogger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings_obj: SettingsType | None = None) -> None:
        self.client = client or LLMClient(settings=settings_obj or settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except (asyncio.TimeoutError, OSError) as e:
            logger.error("test_execution_failed: error=%s", e)
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Include context about what tasks were run
        task_summary = "\n".join([
            f"  - {t.id}: {t.description} ({'completed' if t.status == 'completed' else t.status})"
            for t in tasks
        ])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTasks run:\n{task_summary}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: check if "passed" is in response
            passed = "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(passed=passed, summary=response, failures=[] if passed else [response])

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
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, OSError):
                continue
        return "No test runner found or no tests to run."
