from __future__ import annotations

import asyncio
import json
import os
import shlex
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        results: dict[str, str] = {}
        try:
            results["tests"] = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            results["tests"] = f"Error: {e}"

        try:
            results["lint"] = await self._run_lint()
        except Exception as e:  # noqa: BLE001
            results["lint"] = f"Error: {e}"

        try:
            results["typecheck"] = await self._run_typecheck()
        except Exception as e:  # noqa: BLE001
            results["typecheck"] = f"Error: {e}"

        test_output = "\n\n".join(f"=== {k} ===\n{v}" for k, v in results.items())

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nCheck results:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_command(self, cmd: list[str], timeout: int = 120) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return stdout.decode() + stderr.decode()
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Command timed out after {timeout}s: {' '.join(cmd)}"
        except FileNotFoundError:
            return f"Command not found: {cmd[0]}"

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
        outputs: list[str] = []
        for cmd in candidates:
            out = await self._run_command(cmd)
            if out and "not found" not in out.lower() and out != "No test runner found.":
                outputs.append(f"$ {' '.join(cmd)}\n{out}")
        return "\n".join(outputs) if outputs else "No test runner found."

    async def _run_lint(self) -> str:
        candidates = [
            ["ruff", "check", "."],
            ["ruff", "check"],
            ["flake8"],
            ["eslint", "."],
            ["eslint"],
        ]
        outputs: list[str] = []
        for cmd in candidates:
            out = await self._run_command(cmd)
            if out and "not found" not in out.lower():
                outputs.append(f"$ {' '.join(cmd)}\n{out}")
        return "\n".join(outputs) if outputs else "No linter found."

    async def _run_typecheck(self) -> str:
        candidates = [
            ["mypy", "."],
            ["mypy"],
            ["tsc", "--noEmit"],
            ["tsc"],
        ]
        outputs: list[str] = []
        for cmd in candidates:
            out = await self._run_command(cmd)
            if out and "not found" not in out.lower():
                outputs.append(f"$ {' '.join(cmd)}\n{out}")
        return "\n".join(outputs) if outputs else "No type checker found."
