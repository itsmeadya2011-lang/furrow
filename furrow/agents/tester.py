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


logger = structlog.get_logger(__name__)


class TesterAgent:
    DEFAULT_TIMEOUT = 120

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    def _get_test_commands(self) -> list[list[str]]:
        override = getattr(self.client.settings, "test_commands", None)
        if override:
            return override
        return [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - top-level safety net for test execution
            logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        logger.debug("evaluating_test_output", goal=goal)
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                raise TypeError("Expected a JSON object from tester response")
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug("tester_response_not_valid_json", error=str(e))
            passed = self._extract_bool(response)
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        timeout = getattr(self.client.settings, "tester_timeout", self.DEFAULT_TIMEOUT)
        commands = self._get_test_commands()
        logger.debug("test_commands_available", commands=commands)
        for cmd in commands:
            logger.debug("trying_test_command", command=cmd)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except Exception as e:  # noqa: BLE001 - ignore commands that aren't installed
                logger.debug("test_command_not_found", command=cmd, error=str(e))
                continue
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.debug("test_command_timed_out", command=cmd, timeout=timeout)
                proc.kill()
                continue
            output = stdout.decode() + stderr.decode()
            if proc.returncode != 0:
                logger.debug("test_command_failed", command=cmd, returncode=proc.returncode)
                output += f"\n[exited with code {proc.returncode}]\n"
            else:
                logger.debug("test_runner_found", command=cmd, returncode=proc.returncode)
            return output
        logger.warning("no_test_runner_found")
        return "No test runner found."

    @staticmethod
    def _extract_bool(text: str) -> bool:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    for key in ("passed", "pass", "success"):
                        value = data.get(key)
                        if isinstance(value, bool):
                            return value
            except (json.JSONDecodeError, ValueError):
                pass
        lower = text.lower()
        return "passed" in lower
