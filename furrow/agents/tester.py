from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("tester_start", tasks=len(tasks))
        # Run all checks
        test_output = await self._run_tests()
        lint_output = await self._run_lint()
        type_output = await self._run_type_check()

        combined_output = f"=== Tests ===\n{test_output}\n\n=== Lint ===\n{lint_output}\n\n=== Type Check ===\n{type_output}"

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{combined_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester_complete", passed=result.passed, failures=len(result.failures))
            return result
        except (json.JSONDecodeError, ValueError):
            # Heuristic: check if any failures in output
            passed = (
                "passed" in test_output.lower()
                and "failed" not in test_output.lower()
                and "error" not in test_output.lower()
            )
            logger.warning("tester_parse_failed", response=response[:200], heuristic_passed=passed)
            return TestResult(passed=passed, summary=response, failures=[])

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
                    logger.debug("test_run", cmd=cmd[0], output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", cmd=cmd[0])
                    continue
            except FileNotFoundError:
                logger.debug("test_cmd_not_found", cmd=cmd[0])
                continue
            except Exception as e:  # noqa: BLE001
                logger.debug("test_cmd_not_found", cmd=cmd[0], error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."

    async def _run_lint(self) -> str:
        candidates = [
            ["ruff", "check", "."],
            ["flake8", "."],
            ["pylint", "."],
            ["eslint", "."],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        logger.debug("lint_run", cmd=cmd[0], output_length=len(output))
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("lint_timeout", cmd=cmd[0])
                    continue
            except FileNotFoundError:
                logger.debug("lint_cmd_not_found", cmd=cmd[0])
                continue
            except Exception as e:  # noqa: BLE001
                logger.debug("lint_cmd_not_found", cmd=cmd[0], error=str(e))
                continue
        logger.debug("no_linter_found")
        return "No linter found."

    async def _run_type_check(self) -> str:
        candidates = [
            ["mypy", "."],
            ["pyright", "."],
            ["tsc", "--noEmit"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        logger.debug("type_check_run", cmd=cmd[0], output_length=len(output))
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("type_check_timeout", cmd=cmd[0])
                    continue
            except FileNotFoundError:
                logger.debug("type_check_cmd_not_found", cmd=cmd[0])
                continue
            except Exception as e:  # noqa: BLE001
                logger.debug("type_check_cmd_not_found", cmd=cmd[0], error=str(e))
                continue
        logger.debug("no_type_checker_found")
        return "No type checker found."
