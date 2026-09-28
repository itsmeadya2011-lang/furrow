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


class TesterAgent:
    DEFAULT_TEST_COMMANDS: list[list[str]] = [
        ["pytest", "-q"],
        ["python", "-m", "pytest", "-q"],
        ["npm", "test", "--", "--silent"],
        ["pnpm", "test", "--", "--silent"],
        ["yarn", "test", "--silent"],
        ["cargo", "test", "-q"],
        ["go", "test", "./..."],
    ]

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        test_commands: list[list[str]] | None = None,
        timeout: int = 120,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self.test_commands = test_commands or list(self.DEFAULT_TEST_COMMANDS)
        self.timeout = timeout

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
        output_parts: list[str] = []
        tried: list[str] = []
        ran_tests = False

        for cmd in self.test_commands:
            cmd_str = " ".join(cmd)
            tried.append(cmd_str)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self.timeout
                    )
                    output_parts.append(f"[{cmd_str}]:\n" + stdout.decode() + stderr.decode())
                    ran_tests = True
                    break
                except asyncio.TimeoutError:
                    proc.kill()
                    output_parts.append(f"[{cmd_str}] timed out after {self.timeout}s")
                    continue
            except FileNotFoundError:
                continue
            except Exception:
                continue

        if not ran_tests:
            output_parts.append(f"No test runner found. Tried: {', '.join(tried)}")

        lint_output = await self._run_linters()
        if lint_output:
            output_parts.append(lint_output)

        return "\n".join(output_parts)

    async def _run_linters(self) -> str:
        parts: list[str] = []

        if shutil.which("ruff"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ruff",
                    "check",
                    ".",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self.timeout
                    )
                    parts.append(f"[ruff check .]:\n" + stdout.decode() + stderr.decode())
                except asyncio.TimeoutError:
                    proc.kill()
                    parts.append("[ruff check .] timed out")
            except Exception:
                pass

        if shutil.which("mypy"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "python",
                    "-m",
                    "mypy",
                    "furrow/",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self.timeout
                    )
                    parts.append(f"[mypy furrow/]:\n" + stdout.decode() + stderr.decode())
                except asyncio.TimeoutError:
                    proc.kill()
                    parts.append("[mypy furrow/] timed out")
            except Exception:
                pass

        return "\n".join(parts) if parts else ""
