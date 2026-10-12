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
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        return self._parse_response(response, test_output)

    def _parse_response(self, response: str, test_output: str) -> TestResult:
        """Parse the LLM response, trying JSON first, then falling back to heuristics."""
        # Try to extract JSON from the response (handles markdown code fences)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        try:
            data = json.loads(cleaned)
            return TestResult(
                passed=bool(data.get("passed", False)),
                summary=str(data.get("summary", "")),
                failures=list(data.get("failures", [])),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: try to find JSON object anywhere in the response
        json_match = re.search(r"\{[^}]*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return TestResult(
                    passed=bool(data.get("passed", False)),
                    summary=str(data.get("summary", "")),
                    failures=list(data.get("failures", [])),
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Last resort: string matching on the raw output
        passed = "passed" in response.lower() and "fail" not in response.lower()
        return TestResult(passed=passed, summary=response, failures=[])

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