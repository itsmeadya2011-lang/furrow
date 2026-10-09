from __future__ import annotations

import asyncio
import json
import os
import shlex
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


logger = structlog.get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        attempted_commands: list[str] = []
        succeeded_cmd = ""
        try:
            test_output, attempted_commands, succeeded_cmd = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        header = f"Test command: {succeeded_cmd}\n" if succeeded_cmd else ""
        if attempted_commands and not succeeded_cmd:
            header += f"Attempted commands: {', '.join(attempted_commands)}\n"
        
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n{header}Test output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _build_candidates(self) -> list[list[str]]:
        candidates: list[list[str]] = []
        
        # Check environment variable first
        env_cmd = os.environ.get("FURROW_TEST_CMD")
        if env_cmd:
            try:
                candidates.append(shlex.split(env_cmd))
            except ValueError:
                logger.warning("invalid_furrow_test_cmd", cmd=env_cmd)
        
        # Check settings for custom test command
        if hasattr(self.settings, "test_command") and self.settings.test_command:
            try:
                candidates.append(shlex.split(self.settings.test_command))
            except ValueError:
                logger.warning("invalid_test_command_setting", cmd=self.settings.test_command)
        
        # Default auto-detection candidates
        candidates.extend([
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ])
        
        return candidates

    async def _run_tests(self) -> tuple[str, list[str], str]:
        candidates = self._build_candidates()
        attempted_commands: list[str] = []
        timeout = getattr(self.settings, "test_timeout", 120)
        
        for cmd in candidates:
            cmd_str = " ".join(cmd)
            attempted_commands.append(cmd_str)
            logger.debug("attempting_test_command", command=cmd_str)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        logger.debug("test_command_succeeded", command=cmd_str)
                        return output, attempted_commands, cmd_str
                    else:
                        logger.debug("test_command_empty_output", command=cmd_str)
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.debug("test_command_timeout", command=cmd_str, timeout=timeout)
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_command_failed", command=cmd_str, error=str(e))
                continue
        
        # All commands failed or returned empty output
        output = "No test runner found or all commands returned empty output."
        if attempted_commands:
            output += f" Attempted: {', '.join(attempted_commands)}"
        return output, attempted_commands, ""
