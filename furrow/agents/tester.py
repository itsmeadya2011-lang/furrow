from __future__ import annotations

import asyncio
import json
import shutil
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


TEST_COMMANDS = [
    ["pytest", "-q"],
    ["python", "-m", "pytest", "-q"],
    ["npm", "test", "--", "--silent"],
    ["pnpm", "test", "--", "--silent"],
    ["yarn", "test", "--silent"],
    ["cargo", "test", "-q"],
    ["go", "test", "./..."],
]

LINT_COMMANDS = [
    ["ruff", "check", "."],
    ["flake8"],
    ["eslint", "."],
]

TYPE_CHECK_COMMANDS = [
    ["mypy", "."],
    ["pyright"],
    ["tsc", "--noEmit"],
]


class TesterAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        outputs: list[str] = []

        test_output = await self._run_command_group(TEST_COMMANDS, "tests")
        outputs.append(f"=== TESTS ===\n{test_output}")

        lint_output = await self._run_command_group(LINT_COMMANDS, "lint")
        outputs.append(f"=== LINT ===\n{lint_output}")

        type_output = await self._run_command_group(TYPE_CHECK_COMMANDS, "type check")
        outputs.append(f"=== TYPE CHECK ===\n{type_output}")

        full_output = "\n\n".join(outputs)

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{full_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower() and "fail" not in response.lower(),
                summary=response,
                failures=[],
            )

    async def _run_command_group(self, commands: list[list[str]], label: str) -> str:
        results: list[str] = []
        for cmd in commands:
            if not shutil.which(cmd[0]):
                continue
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
                        results.append(f"$ {' '.join(cmd)}\n{output}")
                except asyncio.TimeoutError:
                    proc.kill()
                    results.append(f"$ {' '.join(cmd)}\n[TIMEOUT]")
            except Exception:
                continue
        return "\n".join(results) if results else f"No {label} runner found."
