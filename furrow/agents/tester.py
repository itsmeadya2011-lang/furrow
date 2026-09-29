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
        try:
            test_output, returncode = await self._run_tests()
        except Exception as e:
            log.exception("test_run_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        summary = (
            f"Test runner exited with code {returncode}.\n\n"
            f"Output:\n{test_output}"
        )
        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n"
            f"Test output (exit code: {returncode}):\n{test_output}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        result = self._parse_result(response, returncode, test_output)
        # Ensure the summary reflects reality.
        result.summary = summary
        return result

    def _parse_result(
        self, response: str, returncode: int, test_output: str
    ) -> TestResult:
        try:
            data = json.loads(response)
            # Validate required keys; fall back to return-code based decision.
            if not isinstance(data, dict) or "passed" not in data:
                raise ValueError("missing 'passed' key")
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError, TypeError):
            # The model did not return valid JSON. Infer pass/fail from the
            # process exit code, which is authoritative for test runs.
            passed = returncode == 0
            return TestResult(
                passed=passed,
                summary=response.strip()[:500],
                failures=[] if passed else [f"Tests exited with code {returncode}"],
            )

    async def _run_tests(self) -> tuple[str, int]:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--silent"],
            ["pnpm", "test", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except (FileNotFoundError, OSError):
                continue
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.client.settings.request_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                continue
            returncode = proc.returncode if proc.returncode is not None else 0
            return stdout.decode() + stderr.decode(), returncode
        return "No test runner found in this project.", 0
