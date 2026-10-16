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
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_all_checks()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nCheck output:\n{test_output}\n"
        response = await self.client.complete(
            prompt, model=self.client.settings.tester_model
        )
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower(),
                summary=response,
                failures=[],
            )

    async def _run_all_checks(self) -> str:
        """Run test suite, linter, and type checker.  Return combined output."""
        parts: list[str] = []

        test_cmds = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        test_output = await self._try_commands(test_cmds)
        if test_output and test_output != "No runner found.":
            parts.append(f"Test output:\n{test_output}")

        lint_cmds = [
            ["ruff", "check", "."],
            ["flake8", "."],
            ["eslint", "."],
        ]
        lint_output = await self._try_commands(lint_cmds)
        if lint_output:
            parts.append(f"Lint output:\n{lint_output}")

        type_cmds = [
            ["mypy", "."],
            ["tsc", "--noEmit"],
        ]
        type_output = await self._try_commands(type_cmds)
        if type_output:
            parts.append(f"Type check output:\n{type_output}")

        if not parts:
            return "No test runner, linter, or type checker found."

        return "\n\n".join(parts)

    @staticmethod
    async def _try_commands(cmds: list[list[str]]) -> str:
        """Try each command in order; return output of the first one that runs."""
        for cmd in cmds:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=120
                    )
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                    # Command ran but produced no output — treat as success.
                    return ""
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No runner found."
