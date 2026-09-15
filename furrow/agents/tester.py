from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult, settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


@dataclass
class _TestCommandResult:
    command: list[str]
    returncode: int
    output: str
    error: str | None = None


class TesterAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        if settings is not None:
            self.settings = settings
        else:
            client_settings = getattr(client, "settings", None)
            self.settings = (
                client_settings
                if client_settings is not None
                else default_settings
            )
        self.client = client or LLMClient(settings=self.settings)
        self._last_test_execution: _TestCommandResult | None = None

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self._last_test_execution = None
        try:
            test_output = await self._run_tests()
            execution = self._last_test_execution
            if execution is not None and execution.error is not None:
                return TestResult(
                    passed=False,
                    summary=execution.error,
                    failures=[execution.error],
                )
            if execution is not None and execution.returncode != 0:
                command = " ".join(execution.command)
                summary = (
                    f"Test command exited with exit status {execution.returncode}: "
                    f"{command}"
                )
                if execution.output.strip():
                    summary = f"{summary}\n{execution.output}"
                return TestResult(
                    passed=False,
                    summary=summary,
                    failures=[summary],
                )

            prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
            response = await self.client.complete(
                prompt,
                model=self.settings.tester_model,
            )
            try:
                data = json.loads(response)
                return TestResult(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return TestResult(
                    passed=False,
                    summary=response,
                    failures=["Failed to parse tester response as JSON."],
                )
        except Exception as exc:
            summary = f"Test execution failed: {exc}"
            return TestResult(passed=False, summary=summary, failures=[summary])

    async def _run_tests(self) -> str:
        execution = await self._execute_tests()
        self._last_test_execution = execution
        return execution.error or execution.output

    async def _execute_tests(self) -> _TestCommandResult:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        last_error = "No test runner found."
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.settings.workspace,
                )
            except FileNotFoundError:
                continue
            except Exception as exc:
                return _TestCommandResult(
                    command=cmd,
                    returncode=-1,
                    output="",
                    error=f"Failed to start test command {cmd}: {exc}",
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass
                last_error = f"Test command timed out: {' '.join(cmd)}"
                continue

            stdout_text = (
                stdout.decode(errors="replace")
                if isinstance(stdout, (bytes, bytearray))
                else str(stdout)
            )
            stderr_text = (
                stderr.decode(errors="replace")
                if isinstance(stderr, (bytes, bytearray))
                else str(stderr)
            )
            output = stdout_text + stderr_text
            returncode = proc.returncode if proc.returncode is not None else -1
            return _TestCommandResult(
                command=cmd,
                returncode=returncode,
                output=output,
            )

        return _TestCommandResult(
            command=[],
            returncode=-1,
            output="",
            error=last_error,
        )
