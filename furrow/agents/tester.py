from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("tester_run_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        logger.info("tester_test_output", length=len(test_output))

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        except Exception as e:
            logger.error("tester_llm_error", error=str(e))
            return TestResult(passed=False, summary=f"LLM error: {e}", failures=[str(e)])

        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("tester_parse_error", error=str(e), response=response[:200])
            passed = "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(passed=passed, summary=response, failures=[] if passed else [response])

    async def _run_tests(self) -> str:
        # Detect project type and run appropriate tests
        project_type = self._detect_project_type()

        candidates = self._get_test_commands(project_type)

        for cmd in candidates:
            try:
                logger.debug("tester_trying_command", cmd=cmd)
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
                    output = stdout.decode() + stderr.decode()
                    logger.info("tester_command_result", cmd=cmd, returncode=proc.returncode, output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("tester_timeout", cmd=cmd)
                    continue
            except FileNotFoundError:
                logger.debug("tester_command_not_found", cmd=cmd)
                continue
            except Exception as e:
                logger.error("tester_command_error", cmd=cmd, error=str(e))
                continue

        return "No test runner found or all commands failed."

    def _detect_project_type(self) -> str:
        """Detect the project type based on files in the workspace."""
        workspace = Path(self.settings.workspace) if self.settings else Path.cwd()

        if (workspace / "package.json").exists():
            return "node"
        elif (workspace / "Cargo.toml").exists():
            return "rust"
        elif (workspace / "go.mod").exists():
            return "go"
        elif (workspace / "pyproject.toml").exists() or (workspace / "setup.py").exists() or (workspace / "requirements.txt").exists():
            return "python"
        elif (workspace / "pom.xml").exists() or (workspace / "build.gradle").exists():
            return "java"
        return "unknown"

    def _get_test_commands(self, project_type: str) -> list[list[str]]:
        """Get test commands for the detected project type."""
        commands = {
            "python": [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["python", "-m", "unittest", "discover", "-v"],
            ],
            "node": [
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
            ],
            "rust": [
                ["cargo", "test", "-q"],
            ],
            "go": [
                ["go", "test", "./..."],
            ],
            "java": [
                ["./mvnw", "test", "-q"],
                ["mvn", "test", "-q"],
                ["./gradlew", "test", "--quiet"],
                ["gradle", "test", "--quiet"],
            ],
        }
        return commands.get(project_type, [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ])