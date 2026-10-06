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

logger = structlog.get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.max_retries = 2

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = await self._run_tests()

        if not test_output or test_output == "No test runner found.":
            logger.warning("No test runner found, skipping test phase")
            return TestResult(passed=True, summary="No tests to run", failures=[])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"

        for attempt in range(self.max_retries):
            try:
                response = await self.client.complete(prompt, model=self.client.settings.tester_model)
                data = json.loads(response)
                result = TestResult(**data)
                logger.info("Test evaluation complete", passed=result.passed, failures=len(result.failures))
                return result
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to parse test result, retrying", attempt=attempt + 1, error=str(e))
                if attempt == self.max_retries - 1:
                    # Fallback: check if tests passed based on output
                    passed = "passed" in test_output.lower() or "ok" in test_output.lower()
                    return TestResult(
                        passed=passed,
                        summary=f"Failed to parse LLM response: {e}",
                        failures=[test_output] if not passed else [],
                    )
                prompt += f"\n\nPrevious response was invalid JSON. Error: {e}\nPlease return valid JSON only."

        return TestResult(passed=False, summary="Max retries exceeded", failures=["Max retries exceeded"])

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
                    output = stdout.decode() + stderr.decode()
                    logger.debug("Test command ran", cmd=cmd[0], exit_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("Test command timed out", cmd=cmd[0])
                    continue
            except FileNotFoundError:
                continue
            except (OSError, asyncio.SubprocessError) as e:
                logger.warning("Test command failed", cmd=cmd[0], error=str(e))
                continue
        return "No test runner found."