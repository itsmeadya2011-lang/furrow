from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Final

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

TEST_TIMEOUT: Final[int] = 120
NO_TEST_RUNNER: Final[int | None] = None

TEST_COMMANDS: Final[list[list[str]]] = [
    ["pytest", "-q"],
    ["python", "-m", "pytest", "-q"],
    ["npm", "test", "--", "--silent"],
    ["pnpm", "test", "--", "--silent"],
    ["yarn", "test", "--silent"],
    ["cargo", "test", "-q"],
    ["go", "test", "./..."],
]


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            returncode, output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if returncode is None or returncode == NO_TEST_RUNNER:
            return TestResult(
                passed=True, summary="No test runner found; nothing to test.", failures=[]
            )

        if returncode == 0:
            summary = output.strip() or "Test suite exited with code 0."
            return TestResult(passed=True, summary=summary, failures=[])

        return await self._summarize_failures(goal, tasks, output, returncode)

    async def _summarize_failures(
        self, goal: str, tasks: list[TaskModel], output: str, returncode: int
    ) -> TestResult:
        files = self._task_files(tasks)
        context = "\n".join(f"- {path}" for path in files)
        file_context = f"\n\nFiles involved in this goal:\n{context}" if context else ""
        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n"
            f"Test command exit code: {returncode} (non-zero means the suite failed)."
            f"{file_context}\n\nTest output:\n{output}\n"
        )
        summary = output.strip() or f"Test suite exited with code {returncode}."
        try:
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        except Exception:
            return TestResult(passed=False, summary=summary, failures=[])

        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            data = None
        if not isinstance(data, dict):
            return TestResult(passed=False, summary=response.strip() or summary, failures=[])
        return TestResult(
            passed=False,
            summary=str(data.get("summary") or summary),
            failures=[str(failure) for failure in data.get("failures") or []],
        )

    def _task_files(self, tasks: list[TaskModel]) -> list[str]:
        files: dict[str, None] = {}
        for task in tasks:
            for path in task.files:
                files.setdefault(path, None)
        return list(files)

    async def _run_tests(self) -> tuple[int | None, str]:
        cwd = str(self.client.settings.workspace)
        for cmd in TEST_COMMANDS:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            except FileNotFoundError:
                continue
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TEST_TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return (
                    1,
                    f"Command {' '.join(cmd)} timed out after {TEST_TIMEOUT}s.",
                )
            return proc.returncode, stdout.decode(errors="replace") + stderr.decode(
                errors="replace"
            )
        return NO_TEST_RUNNER, "No test runner found."
