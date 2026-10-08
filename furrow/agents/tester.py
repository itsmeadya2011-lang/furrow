from __future__ import annotations

import asyncio
import json
import re
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
        max_iterations = 2  # Try to fix up to once
        for iteration in range(max_iterations):
            test_output = ""
            try:
                test_output = await self._run_tests()
            except Exception as e:
                return TestResult(passed=False, summary=str(e), failures=[str(e)])

            prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
            response = self._strip_json_markers(response)

            try:
                data = json.loads(response)
            except (json.JSONDecodeError, ValueError):
                return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

            # If tests passed, return result
            if data.get("passed", False):
                return TestResult(
                    passed=True,
                    summary=data.get("summary", "All tests passed"),
                    failures=data.get("failures", []),
                )

            # If tests failed and we have fixes, apply them and retry
            if iteration < max_iterations - 1 and data.get("fixes"):
                fixes = data.get("fixes", [])
                for fix in fixes:
                    path = fix.get("path")
                    content = fix.get("content")
                    if path and isinstance(content, str):
                        try:
                            await self.client.write_file(path, content)
                        except Exception as e:
                            return TestResult(passed=False, summary=f"Failed to apply fix: {e}", failures=[str(e)])
                # Continue to retry
                continue

            # No more retries or no fixes, return failure
            return TestResult(
                passed=False,
                summary=data.get("summary", "Tests failed"),
                failures=data.get("failures", []),
            )

        return TestResult(passed=False, summary="Max iterations reached", failures=[])

    def _strip_json_markers(self, text: str) -> str:
        text = text.strip()
        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip() in ("```", "```json", "```JSON"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

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