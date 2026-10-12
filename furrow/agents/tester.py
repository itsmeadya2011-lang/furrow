from __future__ import annotations

import asyncio
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

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            data = await self.client.complete_json(prompt, model=self.client.settings.tester_model)
            return TestResult(**data)
        except ValueError as exc:
            # complete_json embeds the raw response in the error message; reuse it
            # instead of making a second LLM call.
            response = str(exc)
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = [
            ("pytest", ["pytest", "-q"]),
            ("python-pytest", ["python", "-m", "pytest", "-q"]),
            ("npm", ["npm", "test", "--", "--silent"]),
            ("pnpm", ["pnpm", "test", "--", "--silent"]),
            ("yarn", ["yarn", "test", "--silent"]),
            ("cargo", ["cargo", "test", "-q"]),
            ("go", ["go", "test", "./..."]),
        ]
        for runner, cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
                return f"[{runner}] exit={proc.returncode}\n{stdout.decode()}\n{stderr.decode()}"
            except FileNotFoundError:
                continue
        return "No test runner found."