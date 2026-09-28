from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger("tester")

_TEST_COMMANDS = [
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
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            log.error("test_execution_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            log.warning("test_result_parse_fallback", passed=passed)
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        for cmd in _TEST_COMMANDS:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    log.warning("test_timeout", command=" ".join(cmd))
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except Exception:  # noqa: BLE001, S112
                continue
        log.warning("no_test_runner_found")
        return "No test runner found."
