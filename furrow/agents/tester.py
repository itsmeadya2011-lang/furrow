from __future__ import annotations

import asyncio
import json
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
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # If no test runner was found, don't ask the LLM to judge — return
        # explicitly so the orchestrator can decide what to do.
        if test_output.startswith("No test runner found"):
            return TestResult(
                passed=False,
                summary=test_output,
                failures=[test_output],
            )

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower(), summary=response, failures=[]
            )

    async def _run_tests(self) -> str:
        candidates = [
            (["pytest", "-q"], "pytest"),
            (["python", "-m", "pytest", "-q"], "pytest (python -m)"),
            (["npm", "test", "--", "--silent"], "npm test"),
            (["pnpm", "test", "--", "--silent"], "pnpm test"),
            (["yarn", "test", "--silent"], "yarn test"),
            (["cargo", "test", "-q"], "cargo test"),
            (["go", "test", "./..."], "go test"),
        ]
        tried: list[str] = []
        for cmd, label in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    tried.append(f"{label} (timed out)")
                    continue
            except FileNotFoundError:
                tried.append(f"{label} (not installed)")
                continue
            except Exception as e:
                tried.append(f"{label} (error: {e})")
                continue
        return f"No test runner found. Tried: {', '.join(tried) or 'none'}."
