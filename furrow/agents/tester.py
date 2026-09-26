from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

from structlog import get_logger

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class TesterAgent:
    """Validates work by running the real test suite and interpreting the output."""

    # Each entry: (command, display name)
    _TEST_RUNNERS: list[tuple[list[str], str]] = [
        (["pytest", "-q"], "pytest"),
        (["python", "-m", "pytest", "-q"], "python -m pytest"),
        (["cargo", "test", "-q"], "cargo test"),
        (["go", "test", "./..."], "go test"),
        (["npm", "test", "--", "--silent"], "npm test"),
        (["pnpm", "test", "--", "--silent"], "pnpm test"),
        (["yarn", "test", "--silent"], "yarn test"),
    ]

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is not None:
            self.client = client
        elif settings is not None:
            self.client = LLMClient(settings=settings)
        else:
            self.client = LLMClient()

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_runner_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        logger.debug("tester_response", length=len(response))
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            logger.warning("test_result_parse_fallback", response=response[:200])
            return TestResult(
                passed="passed" in response.lower(), summary=response, failures=[]
            )

    async def _run_tests(self) -> str:
        """Run the first available test runner. Returns combined stdout + stderr."""
        import asyncio

        for cmd, name in self._TEST_RUNNERS:
            executable = cmd[0]
            if not shutil.which(executable):
                logger.debug("test_runner_not_found", runner=name)
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    logger.info("test_runner_executed", runner=name, success=proc.returncode == 0)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_runner_timeout", runner=name)
                    continue
            except Exception as e:
                logger.debug("test_runner_exec_error", runner=name, error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."
