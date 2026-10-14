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
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _detect_test_commands(self) -> list[list[str]]:
        workspace = self.client.settings.workspace
        candidates: list[list[str]] = []

        if (
            (workspace / "pyproject.toml").exists()
            or (workspace / "setup.py").exists()
            or (workspace / "setup.cfg").exists()
            or (workspace / "tox.ini").exists()
            or (workspace / "Makefile").exists()
        ):
            candidates.extend([["python", "-m", "pytest", "-q"], ["pytest", "-q"]])
            logger.info("detected_python_project", workspace=str(workspace))

        if (workspace / "package.json").exists():
            candidates.extend(
                [
                    ["npm", "test", "--", "--silent"],
                    ["pnpm", "test", "--", "--silent"],
                    ["yarn", "test", "--silent"],
                ]
            )
            logger.info("detected_node_project", workspace=str(workspace))

        if (workspace / "Cargo.toml").exists():
            candidates.append(["cargo", "test", "-q"])
            logger.info("detected_rust_project", workspace=str(workspace))

        if (workspace / "go.mod").exists() or (workspace / "go.sum").exists():
            candidates.append(["go", "test", "./..."])
            logger.info("detected_go_project", workspace=str(workspace))

        if not candidates:
            logger.warning("no_manifest_found", workspace=str(workspace))
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]

        return candidates

    async def _run_tests(self) -> str:
        candidates = await self._detect_test_commands()
        for cmd in candidates:
            logger.info("trying_test_command", command=" ".join(cmd))
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    logger.info("test_command_succeeded", command=" ".join(cmd))
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_command_timeout", command=" ".join(cmd))
                    continue
            except OSError:
                logger.warning("test_command_failed", command=" ".join(cmd))
                continue
        logger.error("no_test_runner_found")
        return "No test runner found."
