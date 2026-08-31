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

logger = structlog.get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except asyncio.CancelledError:
            return TestResult(passed=False, summary="Test run was cancelled", failures=["Test run was cancelled"])
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
        candidates = [
            ["python", "-m", "pytest"],
            ["python", "-m", "pytest", "-q"],
            ["pytest", "-q"],
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
                    try:
                        stdout, stderr = await asyncio.wait_for(proc.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        logger.warning("timed_out_waiting_for_process_exit", cmd=cmd)
                    try:
                        remaining_stdout, remaining_stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
                        return remaining_stdout.decode() + remaining_stderr.decode()
                    except asyncio.TimeoutError:
                        logger.warning("timed_out_draining_remaining_output", cmd=cmd)
                    continue
            except FileNotFoundError:
                continue
            except asyncio.CancelledError:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise
            except Exception:
                logger.exception("unexpected_error_running_test_command", cmd=cmd)
                continue
        return "No test runner found."
