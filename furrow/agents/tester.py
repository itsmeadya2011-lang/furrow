from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)

MAX_OUTPUT_BYTES = 8192


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("tester_run_tests_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _get_test_commands(self) -> list[list[str]]:
        """Return test commands from settings or env var, falling back to defaults."""
        env_val = os.environ.get("FURROW_TEST_COMMANDS")
        if env_val:
            commands = [c.strip() for c in env_val.split(",") if c.strip()]
            return [c.split() for c in commands]
        configured = self.client.settings.test_commands
        if configured:
            return [c.split() if isinstance(c, str) else list(c) for c in configured]
        return [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]

    async def _run_tests(self) -> str:
        candidates = self._get_test_commands()
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    continue
                stdout_str = stdout.decode(errors="replace")
                stderr_str = stderr.decode(errors="replace")
                combined = self._combine_output(stdout_str, stderr_str)
                if combined:
                    return combined
            except (FileNotFoundError, Exception) as e:
                logger.debug("tester_command_failed", command=" ".join(cmd), error=str(e))
                continue
        return "No test runner found."

    def _combine_output(self, stdout: str, stderr: str) -> str:
        parts: list[str] = []
        if stdout:
            parts.append(f"stdout:\n{self._truncate(stdout)}")
        if stderr:
            parts.append(f"stderr:\n{self._truncate(stderr)}")
        return "\n".join(parts)

    def _truncate(self, text: str) -> str:
        if len(text) <= MAX_OUTPUT_BYTES:
            return text
        half = MAX_OUTPUT_BYTES // 2
        return text[:half] + f"\n... [truncated {len(text) - MAX_OUTPUT_BYTES} bytes] ...\n" + text[-half:]