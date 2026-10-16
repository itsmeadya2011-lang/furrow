from __future__ import annotations

import asyncio
import json
import os
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
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

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
        tried = []
        errors = []
        for cmd in candidates:
            cmd_str = " ".join(cmd)
            tried.append(cmd_str)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    stdout_dec = stdout.decode(errors="replace") if stdout else ""
                    stderr_dec = stderr.decode(errors="replace") if stderr else ""
                    if proc.returncode == 0:
                        return (
                            f"Runner: {cmd_str}\n\n"
                            f"STDOUT:\n{stdout_dec}\n\n"
                            f"STDERR:\n{stderr_dec}\n"
                        )
                    else:
                        return (
                            f"Runner: {cmd_str}\n\n"
                            f"Exit code: {proc.returncode}\n\n"
                            f"STDOUT:\n{stdout_dec}\n\n"
                            f"STDERR:\n{stderr_dec}\n\n"
                            f"Errors encountered: {errors}\n"
                            f"Tried commands: {tried}\n"
                        )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    errors.append(f"Timeout after 120s for: {cmd_str}")
                    continue
            except FileNotFoundError:
                errors.append(f"Command not found: {cmd_str}")
                continue
            except Exception as e:
                errors.append(f"Error running {cmd_str}: {type(e).__name__}: {e}")
                continue
        return (
            f"No test runner succeeded.\n\n"
            f"Tried commands: {tried}\n\n"
            f"Errors encountered: {errors}\n"
        )
