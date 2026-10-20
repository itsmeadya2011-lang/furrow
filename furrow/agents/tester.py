from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_SUMMARY_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)

DEFAULT_TEST_TIMEOUT = 120


@dataclass
class CommandResult:
    runner: str | None
    returncode: int | None
    output: str


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or default_settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        result = await self._run_tests()
        if result.runner is None:
            return TestResult(
                passed=True,
                summary="No test runner found; verification skipped",
                failures=[],
            )
        passed = result.returncode == 0
        # LLM summary, but never override pass/fail
        try:
            prompt = (
                f"{TESTER_SUMMARY_PROMPT}\n\nGoal: {goal}\n"
                f"Exit code: {result.returncode}\nOutput:\n{result.output}\n"
            )
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
            data = json.loads(response)
            summary = data.get("summary", result.output)
            failures = data.get("failures", [])
            return TestResult(passed=passed, summary=summary, failures=failures)
        except Exception as e:  # noqa: BLE001
            log.warning("tester.llm_failed", error=str(e))
            return TestResult(passed=passed, summary=result.output, failures=[])

    async def _run_tests(self) -> CommandResult:
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
            if shutil.which(cmd[0]) is None:
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.client.settings.workspace,
                )
            except FileNotFoundError:
                continue
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=DEFAULT_TEST_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return CommandResult(
                    runner=" ".join(cmd),
                    returncode=-1,
                    output=f"Timed out after {DEFAULT_TEST_TIMEOUT} seconds",
                )
            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            return CommandResult(runner=" ".join(cmd), returncode=proc.returncode, output=output)
        return CommandResult(runner=None, returncode=None, output="No test runner found.")
