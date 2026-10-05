from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.util import parse_json_object

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
        try:
            data = parse_json_object(response)
            return TestResult(**data)
        except ValueError:
            passed = "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(
                passed=passed,
                summary=response[:500],
                failures=[response[:500]],
            )

    async def _run_tests(self) -> str:
        workspace = self.client.settings.workspace
        errors: list[str] = []

        if self.client.settings.test_command:
            cmd = self.client.settings.test_command
            try:
                proc = await asyncio.create_subprocess_exec(
                    "/bin/sh", "-c", cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=120
                    )
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    errors.append(f'"{cmd}" timed out')
            except (FileNotFoundError, OSError):
                pass

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
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=120
                    )
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    errors.append(f"{' '.join(cmd)} timed out")
                    continue
            except (FileNotFoundError, OSError):
                continue
        if errors:
            return "No test runner found.\n" + "\n".join(errors)
        return "No test runner found."
