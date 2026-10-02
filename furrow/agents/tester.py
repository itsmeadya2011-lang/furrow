from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        log.info("running_tester", goal=goal, num_tasks=len(tasks))

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        result = self._parse_test_result_response(response)
        log.info(
            "tester_result",
            passed=result.passed,
            summary=result.summary,
            failures=result.failures,
        )
        return result

    def _parse_test_result_response(self, response: str) -> TestResult:
        cleaned = response.strip()
        fence = re.search(r"```(?:[a-zA-Z]+)?\n(.*)```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()

        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except (json.JSONDecodeError, ValueError):
                data = None
            if isinstance(data, dict) and "passed" in data and "summary" in data:
                try:
                    return TestResult(**data)
                except (TypeError, ValueError):
                    pass

        lower = response.lower()
        if re.search(r"\b(?:fail|failed|error)\b", lower):
            return TestResult(passed=False, summary=response, failures=[])
        if re.search(r"\b(?:pass|passed|success|ok)\b", lower):
            return TestResult(passed=True, summary=response, failures=[])
        return TestResult(passed=False, summary=response, failures=[])

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
            log.debug("trying_test_runner", runner=cmd[0])
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    log.info("test_runner_found", runner=cmd[0], returncode=proc.returncode)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    log.warning("test_runner_timeout", runner=cmd[0])
                    continue
            except OSError:
                continue
        return "No test runner found."
