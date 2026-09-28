from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    """Runs tests and evaluates whether they pass via the LLM."""

    TEST_TIMEOUT = 120

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = (
            f"Goal: {goal}\n\n"
            f"Completed tasks:\n"
            + "\n".join(f"- [{t.id}] {t.description}: {t.result or ''}" for t in tasks)
            + f"\n\nTest output:\n{test_output}\n"
        )
        response = await self.client.complete(
            prompt, system=TESTER_PROMPT, model=self.client.settings.tester_model
        )
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
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
            output = await self._try_run(cmd)
            if output is not None:
                return output
        return "No test runner found."

    async def _try_run(self, cmd: list[str]) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError:
            return None
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.TEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                await asyncio.wait_for(proc.wait(), timeout=5)
            return None
        return stdout.decode() + stderr.decode()
