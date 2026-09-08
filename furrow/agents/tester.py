from __future__ import annotations

import asyncio
import json
import os
import structlog
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("tester_exception", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        logger.info("tester_evaluating", test_output_length=len(test_output))
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester_result", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            logger.info("tester_result_fallback", passed=passed)
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = self._detect_test_runners()
        for cmd in candidates:
            try:
                logger.info("running_test", cmd=" ".join(cmd))
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    logger.info("test_completed", cmd=" ".join(cmd), output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", cmd=" ".join(cmd))
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_skipped", cmd=" ".join(cmd), reason=str(e))
                continue
        return "No test runner found."

    def _detect_test_runners(self) -> list[list[str]]:
        workspace = Path(self.client.settings.workspace)
        candidates: list[list[str]] = []

        if (workspace / "pyproject.toml").exists() or (workspace / "setup.py").exists() or (workspace / "setup.cfg").exists():
            candidates.extend([
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
            ])

        if (workspace / "package.json").exists():
            candidates.extend([
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ])

        if (workspace / "Cargo.toml").exists():
            candidates.append(["cargo", "test", "-q"])

        if (workspace / "go.mod").exists():
            candidates.append(["go", "test", "./..."])

        return candidates
