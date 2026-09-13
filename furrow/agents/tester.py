from __future__ import annotations

import asyncio
import json
import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

logger = structlog.get_logger()

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

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            passed = bool(data.get("passed", False))
            summary = str(data.get("summary", ""))
            failures = list(data.get("failures", []))
            return TestResult(passed=passed, summary=summary, failures=failures)
        except (json.JSONDecodeError, ValueError):
            logger.error("tester_json_parse_failed", response=response[:200])
            return TestResult(
                passed=False,
                summary="Could not parse tester response as JSON",
                failures=[f"Parser error: response was not valid JSON: {response[:200]}"],
            )

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
