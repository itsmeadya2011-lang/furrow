from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings as SettingsType

logger = logging.getLogger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: SettingsType | None = None) -> None:
        self.client = client or LLMClient(settings=settings or Settings())

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        lint_output = ""
        type_output = ""

        try:
            test_output = await self._run_tests()
        except OSError as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Run linting
        try:
            lint_output = await self._run_lint()
        except OSError as e:
            lint_output = f"Lint error: {e}"

        # Run type checking
        try:
            type_output = await self._run_type_check()
        except OSError as e:
            type_output = f"Type check error: {e}"

        combined_output = f"=== TEST OUTPUT ===\n{test_output}\n\n=== LINT OUTPUT ===\n{lint_output}\n\n=== TYPE CHECK OUTPUT ===\n{type_output}"

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{combined_output}\n"
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
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except (OSError, ValueError, RuntimeError) as e:
                logger.debug("test_runner_error", extra={"cmd": cmd, "error": str(e)})
                continue
        return "No test runner found."

    async def _run_lint(self) -> str:
        candidates = [
            ["ruff", "check", "."],
            ["flake8", "."],
            ["pylint", "."],
            ["eslint", "."],
            ["golangci-lint", "run"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except (OSError, ValueError, RuntimeError) as e:
                logger.debug("lint_error", extra={"cmd": cmd, "error": str(e)})
                continue
        return "No linter found or no issues."

    async def _run_type_check(self) -> str:
        candidates = [
            ["mypy", "."],
            ["pyright", "."],
            ["tsc", "--noEmit"],
            ["go", "vet", "./..."],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except (OSError, ValueError, RuntimeError) as e:
                logger.debug("type_check_error", extra={"cmd": cmd, "error": str(e)})
                continue
        return "No type checker found or no issues."
