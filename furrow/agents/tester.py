from __future__ import annotations

import asyncio
import json
import os
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
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        data = response
        for attempt in range(1, 4):
            try:
                parsed = json.loads(data)
                return TestResult(**parsed)
            except (json.JSONDecodeError, ValueError):
                if attempt < 3:
                    data = response + f"\n\nAttempt {attempt}/3: Please respond with valid JSON only."
                else:
                    summary = response or "No valid JSON response from model."
                    failures: list[str] = []
                    lower = summary.lower()
                    if "assert" in lower or "error" in lower or "fail" in lower:
                        failures = [line.strip() for line in summary.splitlines() if any(k in line.lower() for k in ("assert", "error", "fail", "traceback"))][:10]
                    return TestResult(passed=False, summary=summary, failures=failures)

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["python", "-m", "unittest", "discover", "-v"],
            ["dotnet", "test", "--no-build"],
            ["mvn", "test", "-q"],
            ["gradle", "test", "--quiet"],
            ["bun", "test"],
            ["deno", "test"],
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
                    return stdout.decode() + stderr.decode() + f"\n[exit code: {proc.returncode}]"
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
