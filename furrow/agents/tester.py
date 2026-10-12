from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class NoTestRunnerError(Exception):
    pass


def _extract_json(text: str) -> str:
    """Strip markdown fences and slice from first '{' to last '}'."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence and optional language tag
        end = text.find("\n")
        if end != -1:
            text = text[end + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_output, passed = await self._run_tests()
        except NoTestRunnerError as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Test output:\n{test_output}\n\n"
            f"The subprocess reported tests as {'PASSED' if passed else 'FAILED'}."
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(_extract_json(response))
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> tuple[str, bool]:
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
            except FileNotFoundError:
                # Runner binary not installed; try the next candidate.
                continue
            except Exception:
                # Unexpected spawn error; log and move on.
                continue
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()
                continue
            output = (stdout.decode() + stderr.decode()).strip()
            return output, proc.returncode == 0
        raise NoTestRunnerError("No test runner found.")