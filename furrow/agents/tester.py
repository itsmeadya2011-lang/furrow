from __future__ import annotations

import asyncio
import shutil
from typing import TYPE_CHECKING, ClassVar

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)

TEST_TIMEOUT_SECONDS = 120

TEST_RUNNER_CANDIDATES: list[list[str]] = [
    ["pytest", "-q"],
    ["python", "-m", "pytest", "-q"],
    ["npm", "test", "--", "--silent"],
    ["pnpm", "test", "--", "--silent"],
    ["yarn", "test", "--silent"],
    ["cargo", "test", "-q"],
    ["go", "test", "./..."],
]

FAILURE_MARKERS = ("fail", "error", "traceback", "not passed")


class TesterAgent:
    # Opt out of pytest collection: the name matches its default "Test*" pattern,
    # but this is the tester agent, not a test class.
    __test__: ClassVar[bool] = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_output = await self._run_tests()
        except OSError as e:
            log.error("tester.run_tests_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = extract_json(response)
            return TestResult(**data)
        except (TypeError, ValueError):
            return TestResult(
                passed=self._looks_passed(response),
                summary=response,
                failures=[],
            )

    @staticmethod
    def _looks_passed(response: str) -> bool:
        lowered = response.lower()
        if any(marker in lowered for marker in FAILURE_MARKERS):
            return False
        return "passed" in lowered

    async def _run_tests(self) -> str:
        available = [cmd for cmd in TEST_RUNNER_CANDIDATES if shutil.which(cmd[0])]
        if not available:
            return "No test runner found."

        last_failure = ""
        for cmd in available:
            command = " ".join(cmd)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except OSError as e:
                log.warning("tester.runner_spawn_failed", command=command, error=str(e))
                continue

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=TEST_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                log.warning("tester.runner_timeout", command=command, timeout=TEST_TIMEOUT_SECONDS)
                proc.kill()
                await proc.wait()
                continue

            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            exit_code = proc.returncode
            if exit_code == 0:
                return f"$ {command} exit={exit_code}\n{output}"
            log.info("tester.runner_failed", command=command, exit_code=exit_code)
            last_failure = f"$ {command} exit={exit_code}\n{output}"

        return last_failure or "No test runner found."