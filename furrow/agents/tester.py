from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
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
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            lowered = response.lower()
            passed = (
                "all tests passed" in lowered
                or "passing" in lowered and "fail" not in lowered
                or "passed" in lowered
            )
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = self.client.settings.workspace
        timeout = self.client.settings.test_timeout

        candidates = []

        makefile = workspace / "Makefile"
        if makefile.exists():
            for line in makefile.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("test:"):
                    candidates.append(["make", "test"])
                    break

        pyproject = workspace / "pyproject.toml"
        if pyproject.exists():
            text = pyproject.read_text()
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("test = "):
                    candidates.append(["python", "-m", "pytest", "-q"])
                    break

        candidates.extend(
            [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]
        )

        for cmd in candidates:
            try:
                logger.info("Attempting test command: %s", " ".join(cmd))
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    output = stdout.decode() + stderr.decode()
                    logger.info("Test output (%s):\n%s", " ".join(cmd), output)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("Test command timed out after %ds: %s", timeout, " ".join(cmd))
                    continue
            except (FileNotFoundError, Exception) as exc:
                logger.debug("Test command failed: %s (%s)", " ".join(cmd), exc)
                continue

        message = (
            "No test runner found. Ensure the project has a test suite and a supported "
            "test command (pytest, npm, cargo, go, make test, etc.)."
        )
        logger.info(message)
        return message
