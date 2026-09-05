from __future__ import annotations

import asyncio
import json

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if not test_output or "No test runner found" in test_output:
            return TestResult(
                passed=True,
                summary="No tests found or no test runner available",
                failures=[],
            )

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            lower = response.lower()
            passed = any(token in lower for token in ("passed", "success", "all tests passed"))
            failed = any(token in lower for token in ("failed", "failure"))
            if failed and not passed:
                return TestResult(passed=False, summary=response, failures=[])
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        primary = self.client.settings.get_test_command()
        fallbacks = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
        ]
        candidates = [primary] + [c for c in fallbacks if c != primary]

        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode == 0:
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    continue
            except FileNotFoundError:
                continue
        return "No test runner found."
