from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from tenacity import retry, retry_if_exception_type, stop_after_attempt

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self._last_response: str = ""

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        await asyncio.sleep(0.5)
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - best-effort: surface test-runner failure as a TestResult
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            return await self._complete_and_parse(prompt)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in self._last_response.lower(),
                summary=self._last_response,
                failures=[],
            )

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(json.JSONDecodeError),
        reraise=True,
    )
    async def _complete_and_parse(self, prompt: str) -> TestResult:
        self._last_response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        data = json.loads(self._last_response)
        return TestResult(**data)

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
                    output = f"Command: {' '.join(cmd)} (exit code: {proc.returncode})\n"
                    return output + stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except Exception:  # noqa: BLE001, S112 - best-effort: skip runners that fail to spawn
                continue
        return "No test runner found."
