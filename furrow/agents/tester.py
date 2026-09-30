from __future__ import annotations

import asyncio
import json
import logging

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

logger = logging.getLogger("furrow")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or Settings())

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        returncode = 0
        try:
            test_output, returncode = await self._run_tests()
        except Exception as e:  # noqa: BLE001 — defensive: test execution must never crash the loop
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        passed = returncode == 0
        failures: list[str] = []
        if returncode != 0:
            failures.append(f"Test runner exited with non-zero status: {returncode}")

        summary = response
        try:
            data = json.loads(response)
            passed = passed and bool(data.get("passed", False))
            summary = data.get("summary", response)
            failures.extend(data.get("failures", []))
        except (json.JSONDecodeError, ValueError):
            passed = passed and ("passed" in response.lower())
        return TestResult(passed=passed, summary=summary, failures=failures)

    async def _run_tests(self) -> tuple[str, int]:
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
            except FileNotFoundError:
                continue
            except OSError as e:
                logger.warning("Failed to launch test runner %s: %s", cmd[0], e)
                continue
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
                return stdout.decode() + stderr.decode(), proc.returncode or 0
            except asyncio.TimeoutError:
                proc.kill()
                continue
        return "No test runner found.", -1
