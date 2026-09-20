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
        except (OSError, asyncio.TimeoutError) as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = self.client.settings.workspace
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        errors: list[str] = []
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                errors.append(f"Command not found: {cmd[0]}")
                continue
            except OSError as e:
                errors.append(f"Failed to start {' '.join(cmd)}: {e}")
                continue

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    await proc.communicate()
                except OSError:
                    pass
                errors.append(f"Timed out: {' '.join(cmd)}")
                continue

            except OSError as e:
                errors.append(f"Failed to read {' '.join(cmd)} output: {e}")
                continue

            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            return output

        if errors:
            return "\n".join(errors)
        return "No test runner found."
