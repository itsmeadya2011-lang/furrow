from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.base import Agent
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger
from furrow.tools.base import Tool

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger(__name__)


class TesterAgent(Agent):
    """Tester agent that runs the project's test suite and evaluates results.

    Attempts to find and run a test runner (pytest, npm, cargo, go, etc.),
    then uses the LLM to interpret the test output and determine whether
    the goal has been met.
    """

    system_prompt = TESTER_PROMPT
    tools: list[type[Tool]] = []  # Tester only needs the LLM, not file tools

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        super().__init__(client=client, settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            log.error("Test runner failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        task_results = "\n".join(
            f"- Task {t.id} ({t.status}): {t.result or 'no result'}"
            for t in tasks
        )
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Goal: {goal}\n\n"
            f"Task results:\n{task_results}\n\n"
            f"Test output:\n{test_output}\n"
        )
        model = self.client.settings.tester_model
        response = await self.client.complete(prompt, model=model)

        # Strip markdown code fences if the LLM wrapped the JSON
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines)

        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower(),
                summary=response,
                failures=[],
            )

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["poetry", "run", "pytest", "-q"],
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
