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
        test_output = ""
        try:
            test_output = await self._run_all_checks()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_command(self, cmd: list[str]) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except FileNotFoundError:
            return f"Command not found: {cmd[0]}"
        except Exception as e:
            return f"Error running {cmd[0]}: {e}"

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.client.settings.test_timeout
            )
            return stdout.decode() + stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Command timed out after {self.client.settings.test_timeout}s: {' '.join(cmd)}"

    async def _run_all_checks(self) -> str:
        sections: list[str] = []

        test_candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        test_output = "No test runner found."
        for cmd in test_candidates:
            output = await self._run_command(cmd)
            if not output.startswith("Command not found"):
                test_output = output
                break
        sections.append(f"=== TESTS ===\n{test_output}")

        if self.client.settings.lint_enabled:
            lint_cmd = list(self.client.settings.lint_command)
            lint_output = await self._run_command(lint_cmd)
            sections.append(f"=== LINT ===\n{lint_output}")

        if self.client.settings.type_check_enabled:
            typecheck_cmd = list(self.client.settings.type_check_command)
            typecheck_output = await self._run_command(typecheck_cmd)
            sections.append(f"=== TYPE CHECK ===\n{typecheck_output}")

        return "\n\n".join(sections)

    async def _run_tests(self) -> str:
        return await self._run_all_checks()
