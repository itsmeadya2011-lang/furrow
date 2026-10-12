from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_INLINE_RE = re.compile(r"(\{[^{}]*\"passed\"[^{}]*\})", re.DOTALL)


def _extract_json(text: str) -> str | None:
    """Try to find a JSON object containing a 'passed' key in the response."""
    # Try fenced code block first
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    # Try inline JSON
    match = _JSON_INLINE_RE.search(text)
    if match:
        return match.group(1)
    # Fallback: find first { ... } pair
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


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

        json_str = _extract_json(response)
        if json_str:
            try:
                data = json.loads(json_str)
                return TestResult(**data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Last-resort heuristic
        lowered = response.lower()
        passed = "passed" in lowered and "fail" not in lowered.split("passed")[0]
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
