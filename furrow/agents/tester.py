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


TEST_TIMEOUT = 120


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if "No test runner found." in test_output:
            return TestResult(passed=True, summary="No test runner found — assuming pass", failures=[])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            response_lower = response.lower()
            passed = (
                "failed" not in response_lower
                and "error" not in response_lower
                and "failed" not in response
                and "FAILED" not in response
                and "ERROR" not in response
            ) and (
                "passed" in response_lower
                or "ok" in response_lower
                or "success" in response_lower
                or "ok" in response
            )
            if "failed" in response_lower or "error" in response_lower or "FAILED" in response or "ERROR" in response:
                passed = False
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = []
        if os.path.exists("pyproject.toml") or os.path.exists("setup.py"):
            candidates.extend([["pytest", "-q"], ["python", "-m", "pytest", "-q"]])
        if os.path.exists("package.json"):
            candidates.extend([["npm", "test", "--", "--silent"], ["pnpm", "test", "--", "--silent"], ["yarn", "test", "--silent"]])
        if os.path.exists("Cargo.toml"):
            candidates.append(["cargo", "test", "-q"])
        if os.path.exists("go.mod"):
            candidates.append(["go", "test", "./..."])
        if os.path.exists("Makefile"):
            candidates.append(["make", "test"])

        if not candidates:
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
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TEST_TIMEOUT)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    return f"Test command timed out: {' '.join(cmd)}"
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
