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


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("test_run_start", goal=goal, task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            logger.info("test_result_parsed", passed=data.get("passed"))
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            logger.warning("test_result_fallback", passed=passed, response_preview=response[:200])
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        timeout = self.settings.test_timeout if self.settings else 120
        candidates = self._detect_test_runners()
        logger.info("test_runners_detected", count=len(candidates), runners=[c[0] for c in candidates])
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    output = stdout.decode() + stderr.decode()
                    logger.info("test_runner_succeeded", runner=cmd[0], output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    logger.warning("test_runner_timeout", runner=cmd[0], timeout=timeout)
                    proc.kill()
                    continue
            except FileNotFoundError:
                logger.debug("test_runner_not_found", runner=cmd[0])
                continue
            except Exception as e:
                logger.debug("test_runner_error", runner=cmd[0], error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."

    def _detect_test_runners(self) -> list[list[str]]:
        """Detect project type and prioritize the appropriate test runner."""
        workspace = self.settings.workspace if self.settings else os.getcwd()
        has_pyproject = os.path.exists(os.path.join(str(workspace), "pyproject.toml"))
        has_setup_py = os.path.exists(os.path.join(str(workspace), "setup.py"))
        has_package_json = os.path.exists(os.path.join(str(workspace), "package.json"))
        has_cargo_toml = os.path.exists(os.path.join(str(workspace), "Cargo.toml"))
        has_go_mod = os.path.exists(os.path.join(str(workspace), "go.mod"))

        if has_pyproject or has_setup_py:
            return [["pytest", "-q"], ["python", "-m", "pytest", "-q"]]
        if has_cargo_toml:
            return [["cargo", "test", "-q"]]
        if has_go_mod:
            return [["go", "test", "./..."]]
        if has_package_json:
            return [["npm", "test", "--", "--silent"], ["pnpm", "test", "--", "--silent"], ["yarn", "test", "--silent"]]

        # Fallback: try all common runners in a sensible order
        return [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]