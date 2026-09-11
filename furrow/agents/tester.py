from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger("furrow.tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            log.error("test_execution_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        return self._parse_result(response, test_output)

    def _parse_result(self, response: str, test_output: str) -> TestResult:
        """Parse the LLM response into a TestResult, handling markdown-wrapped JSON."""
        # Try direct JSON parse
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return TestResult(**data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Try finding any JSON-like object
        brace_match = re.search(r"\{[\s\S]*\}", response)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return TestResult(**data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: infer from response text
        passed = "passed" in response.lower() and "fail" not in response.lower()
        return TestResult(passed=passed, summary=response[:500], failures=[])

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
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode == 0 or "no tests" in output.lower():
                        return output
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    continue
            except (FileNotFoundError, OSError):
                continue
        return "No test runner found."