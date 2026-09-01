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
        logger.info("tester.start", goal=goal)
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("tester.error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if not test_output.strip():
            test_output = "No test output produced."

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        # Try direct JSON parse first, then extract from code block
        data = None
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            cleaned = self._extract_json_block(response)
            if cleaned:
                try:
                    data = json.loads(cleaned)
                except (json.JSONDecodeError, ValueError):
                    pass

        if data is not None:
            try:
                return TestResult(**data)
            except (TypeError, ValueError):
                pass  # Schema mismatch, fall through to text inference

        # Fallback: infer pass/fail from response text
        passed = "passed" in response.lower() and "fail" not in response.lower().split("passed")[0][-20:]
        return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        project_type = self._detect_project_type()
        commands = self._test_commands(project_type)

        for cmd in commands:
            logger.debug("tester.run_command", cmd=" ".join(cmd))
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode == 0 or output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("tester.timeout", cmd=" ".join(cmd))
                    continue
            except FileNotFoundError:
                logger.debug("tester.command_not_found", cmd=" ".join(cmd))
                continue
            except Exception as e:
                logger.warning("tester.command_error", cmd=" ".join(cmd), error=str(e))
                continue

        return "No test runner found."

    def _detect_project_type(self) -> str:
        """Detect the project type based on lock files and config files."""
        cwd = self.client.settings.workspace
        if (cwd / "pyproject.toml").exists():
            return "python"
        if (cwd / "package-lock.json").exists() or (cwd / "pnpm-lock.yaml").exists():
            return "node"
        if (cwd / "Cargo.toml").exists():
            return "rust"
        if (cwd / "go.mod").exists():
            return "go"
        return "unknown"

    def _test_commands(self, project_type: str) -> list[list[str]]:
        """Return prioritized test commands based on project type."""
        candidates: list[list[str]] = []
        if project_type == "python":
            candidates = [
                ["python", "-m", "pytest", "-q"],
                ["pytest", "-q"],
            ]
        elif project_type == "node":
            candidates = [
                ["pnpm", "test", "--silent"],
                ["npm", "test", "--silent"],
                ["yarn", "test", "-s"],
            ]
        elif project_type == "rust":
            candidates = [["cargo", "test", "-q"]]
        elif project_type == "go":
            candidates = [["go", "test", "./..."]]
        else:
            # Fallback: try everything
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--silent"],
                ["pnpm", "test", "--silent"],
                ["yarn", "test", "-s"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]
        return candidates

    @staticmethod
    def _extract_json_block(text: str) -> str | None:
        """Extract JSON from a markdown code block if present."""
        import re

        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
