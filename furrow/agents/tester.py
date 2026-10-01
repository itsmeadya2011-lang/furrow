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

log = structlog.get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            log.error("test_runner_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Robust fallback: only treat as passed if the response explicitly
            # says "passed": true / "false" rather than substring matching.
            lowered = response.lower()
            if '"passed"' in lowered:
                passed = "true" in lowered.split('"passed"')[1][:20]
            elif "all tests passed" in lowered:
                passed = True
            elif "tests failed" in lowered or "tests failed:" in lowered:
                passed = False
            else:
                passed = False
            log.warning(
                "tester_json_parse_fallback",
                passed=passed,
                response_preview=response[:200],
            )
            return TestResult(passed=passed, summary=response[:500], failures=[])

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
                    log.warning("test_command_timeout", command=" ".join(cmd))
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:  # noqa: BLE001
                log.warning("test_command_error", command=" ".join(cmd), error=str(e))
                continue
        return "No test runner found."
