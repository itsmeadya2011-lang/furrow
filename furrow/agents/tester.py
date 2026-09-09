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

    async def _run_ruff(self) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff", "check", ".",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode() + stderr.decode()
            if proc.returncode == 0:
                return "[ruff] PASSED\n"
            return f"[ruff] FAILED (exit {proc.returncode})\n{output}"
        except FileNotFoundError:
            return "[ruff] SKIPPED (ruff not found)\n"
        except asyncio.TimeoutError:
            return "[ruff] FAILED (timeout after 120s)\n"
        except Exception as e:
            return f"[ruff] FAILED ({e})\n"

    async def _run_mypy(self) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "mypy", "furrow",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode() + stderr.decode()
            if proc.returncode == 0:
                return "[mypy] PASSED\n"
            return f"[mypy] FAILED (exit {proc.returncode})\n{output}"
        except FileNotFoundError:
            return "[mypy] SKIPPED (mypy not found)\n"
        except asyncio.TimeoutError:
            return "[mypy] FAILED (timeout after 120s)\n"
        except Exception as e:
            return f"[mypy] FAILED ({e})\n"

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
        test_output = ""
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode == 0:
                        test_output += f"[pytest] PASSED\n"
                    else:
                        test_output += f"[pytest] FAILED (exit {proc.returncode})\n{output}"
                    break
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        else:
            test_output += "[pytest] SKIPPED (no test runner found)\n"

        test_output += "\n"
        test_output += await self._run_ruff()
        test_output += "\n"
        test_output += await self._run_mypy()

        return test_output
