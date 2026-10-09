from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or Settings())

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("tester_started", task_count=len(tasks))

        test_output = ""
        try:
            test_output = await self._run_tests()
        except (asyncio.CancelledError, RuntimeError, OSError) as e:
            logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester_completed", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("tester_parse_failed", error=str(e), response=response)
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])
        except Exception as e:  # noqa: BLE001
            logger.error("tester_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

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
                    logger.debug("test_runner_found", cmd=cmd[0], output_len=len(output))
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", cmd=cmd[0])
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning("test_runner_error", cmd=cmd[0], error=str(e))
                continue
        return "No test runner found."
