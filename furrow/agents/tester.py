from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.agents.prompts import TESTER_PROMPT
from furrow.agents.utils import extract_json
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except (OSError, RuntimeError) as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            return await self._analyze(prompt)
        except ValueError:
            # Fall back to a heuristic if the LLM cannot be parsed after retries.
            return TestResult(passed="passed" in test_output.lower(), summary=test_output, failures=[])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ValueError),
        reraise=True,
    )
    async def _analyze(self, prompt: str) -> TestResult:
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(extract_json(response))
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse test result from LLM: {e}\nResponse: {response}")

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
            except (FileNotFoundError, OSError) as e:
                    logger.debug("Test runner %s unavailable: %s", cmd, e)
                    continue
        return "No test runner found."