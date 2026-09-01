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
    """Runs the project's test suite (and optional lint/type-check), then
    asks the LLM to summarize failures as JSON.
    """

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        from furrow.config import settings as default_settings

        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_output = await self._run_tests()
            lint_output = await self._run_lint()
        except Exception as exc:  # noqa: BLE001
            return TestResult(passed=False, summary=str(exc), failures=[str(exc)])

        combined = (
            f"=== Test output ===\n{test_output}\n"
            f"=== Lint output ===\n{lint_output}\n"
        )
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nOutput:\n{combined}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower(),
                summary=response,
                failures=[],
            )

    async def _exec(self, cmd: list[str], timeout: int) -> tuple[int, str, str] | None:
        if not cmd or shutil.which(cmd[0]) is None:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError):
            return None
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, "", f"Command timed out after {timeout}s: {' '.join(cmd)}"
        return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    async def _run_tests(self) -> str:
        timeout = max(5, int(self.settings.test_timeout_seconds))
        candidates = [
            ["pytest", "-q", "--timeout=30"],
            ["python", "-m", "pytest", "-q", "--timeout=30"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
            result = await self._exec(cmd, timeout)
            if result is None:
                continue
            code, out, err = result
            return f"$ {' '.join(cmd)}\n(exit {code})\n{out}{err}"
        return "No test runner found."

    async def _run_lint(self) -> str:
        timeout = max(5, int(self.settings.test_timeout_seconds))
        candidates = [
            ["ruff", "check", "."],
            ["flake8", "."],
            ["mypy", "."],
            ["eslint", "."],
        ]
        chunks: list[str] = []
        for cmd in candidates:
            result = await self._exec(cmd, timeout)
            if result is None:
                continue
            code, out, err = result
            chunks.append(f"$ {' '.join(cmd)}\n(exit {code})\n{out}{err}")
        return "\n".join(chunks) if chunks else "No linter found."
