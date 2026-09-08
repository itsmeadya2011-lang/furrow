from __future__ import annotations

import asyncio
import json
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_llm(self, prompt: str, model: str) -> str:
        return await self.client.complete(prompt, model=model)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            response = await self._call_llm(prompt, model=self.client.settings.tester_model)
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("tester_parse_failed", error=str(e))
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])
        except Exception as e:
            logger.error("tester_llm_failed", error=str(e))
            return TestResult(passed=False, summary=f"LLM call failed: {e}", failures=[str(e)])

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
                    logger.info("tests_completed", cmd=" ".join(cmd), output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    logger.warning("test_timeout", cmd=" ".join(cmd))
                    proc.kill()
                    continue
            except FileNotFoundError:
                logger.debug("test_runner_not_found", cmd=" ".join(cmd))
                continue
            except Exception as e:
                logger.error("test_runner_error", cmd=" ".join(cmd), error=str(e))
                raise
        logger.warning("no_test_runner_found")
        return "No test runner found."
