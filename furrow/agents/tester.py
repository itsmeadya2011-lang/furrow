from __future__ import annotations

from pathlib import Path

import asyncio
import json
import os
import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

logger = structlog.get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
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
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = self.settings.workspace if self.settings else Path.cwd()
        candidates: list[list[str]] = []

        if (workspace / "pyproject.toml").exists():
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
            ]
        elif (workspace / "package.json").exists():
            candidates = [
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ]
        elif (workspace / "Cargo.toml").exists():
            candidates = [["cargo", "test", "-q"]]
        elif (workspace / "go.mod").exists():
            candidates = [["go", "test", "./..."]]
        else:
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]

        timeout = getattr(self.settings, "test_timeout", 120)

        for cmd in candidates:
            logger.info("running_test_command", command=" ".join(cmd))
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    output = stdout.decode() + stderr.decode()
                    logger.info("test_command_completed", command=" ".join(cmd), returncode=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_command_timed_out", command=" ".join(cmd), timeout=timeout)
                    continue
            except FileNotFoundError:
                logger.debug("test_command_not_found", command=" ".join(cmd))
                continue
            except Exception as e:
                logger.error("test_command_failed", command=" ".join(cmd), error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."
