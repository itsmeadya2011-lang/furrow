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
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
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
        last_error = ""
        for cmd in candidates:
            cmd_str = " ".join(cmd)
            print(f"Attempting test runner: {cmd_str}")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except FileNotFoundError:
                last_error = f"Command not found: {cmd_str}"
                continue
            except OSError as e:
                last_error = f"Failed to start {cmd_str}: {e}"
                continue

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode() + stderr.decode()
                if proc.returncode != 0:
                    return f"Test runner {cmd_str} exited with code {proc.returncode}:\n{output}"
                return output
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    stdout, stderr = await proc.communicate()
                    output = stdout.decode() + stderr.decode()
                    last_error = f"Test runner {cmd_str} timed out after 120s:\n{output}"
                except OSError as e:
                    last_error = f"Test runner {cmd_str} timed out and failed to collect output: {e}"
                continue
            except OSError as e:
                last_error = f"Test runner {cmd_str} failed: {e}"
                continue

        return f"No test runner succeeded. Last error: {last_error}"
