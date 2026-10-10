from __future__ import annotations

import asyncio
import json
import shutil
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
            logger.info("tester.run.start", goal=goal, tasks=len(tasks))
            test_output = await self._run_all_checks()
        except Exception as e:
            logger.error("tester.run.error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_all_checks(self) -> str:
        logger.info("tester.checks.start")
        sections: list[str] = []

        test_output = await self._run_command(self._find_test_command(), "Tests")
        if test_output is not None:
            sections.append(test_output)

        lint_output = await self._run_command(self._find_lint_command(), "Lint")
        if lint_output is not None:
            sections.append(lint_output)

        typecheck_output = await self._run_command(self._find_typecheck_command(), "Typecheck")
        if typecheck_output is not None:
            sections.append(typecheck_output)

        combined = "\n".join(sections) if sections else "No test runner found."
        logger.info("tester.checks.complete", sections=len(sections))
        return combined

    async def _run_command(self, cmd: list[str] | None, name: str) -> str | None:
        if cmd is None:
            logger.debug("tester.command.skip", name=name)
            return None

        logger.info("tester.command.start", name=name, cmd=cmd)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except (FileNotFoundError, OSError) as e:
            logger.warning("tester.command.launch_failed", name=name, error=str(e))
            return f"=== {name} ===\nFailed to launch command: {e}"

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning("tester.command.timeout", name=name, timeout=120)
            return f"=== {name} ===\nCommand timed out after 120 seconds."

        output = stdout.decode() + stderr.decode()
        logger.debug("tester.command.result", name=name, output=output)
        return f"=== {name} ===\n{output}"

    def _find_test_command(self) -> list[str] | None:
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
            if shutil.which(cmd[0]):
                logger.debug("tester.test_command.found", cmd=cmd)
                return cmd
        logger.warning("tester.test_command.not_found")
        return None

    def _find_lint_command(self) -> list[str] | None:
        if shutil.which("ruff"):
            logger.debug("tester.lint_command.found")
            return ["ruff", "check", "furrow/", "tests/"]
        logger.debug("tester.lint_command.not_found")
        return None

    def _find_typecheck_command(self) -> list[str] | None:
        if shutil.which("mypy"):
            logger.debug("tester.typecheck_command.found")
            return ["mypy", "furrow/"]
        logger.debug("tester.typecheck_command.not_found")
        return None
