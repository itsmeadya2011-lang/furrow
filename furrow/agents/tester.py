from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
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

    @property
    def _timeout(self) -> int:
        return int(getattr(self.client.settings, "test_timeout", 120))

    @property
    def _workspace(self) -> Path:
        return Path(getattr(self.client.settings, "workspace", os.getcwd()))

    def _detect_test_commands(self) -> list[list[str]]:
        workspace = self._workspace
        candidates: list[list[str]] = []

        pyproject = workspace / "pyproject.toml"
        makefile = workspace / "Makefile"
        package_json = workspace / "package.json"
        cargo_toml = workspace / "Cargo.toml"
        go_mod = workspace / "go.mod"

        pyproject_has_pytest = False
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                pyproject_has_pytest = "pytest" in content
            except OSError:
                pyproject_has_pytest = False

        package_json_has_test = False
        if package_json.exists():
            try:
                content = package_json.read_text(encoding="utf-8")
                package_json_has_test = '"test"' in content
            except OSError:
                package_json_has_test = False

        makefile_has_test = False
        if makefile.exists():
            try:
                content = makefile.read_text(encoding="utf-8")
                makefile_has_test = "test" in content
            except OSError:
                makefile_has_test = False

        if pyproject.exists() or (workspace / "setup.py").exists():
            # Prioritize pytest for Python projects.
            if pyproject_has_pytest:
                candidates.append(["python", "-m", "pytest", "-q"])
            candidates.append(["python", "-m", "unittest", "discover", "-s", "tests"])
        elif cargo_toml.exists():
            candidates.append(["cargo", "test", "-q"])
        elif go_mod.exists():
            candidates.append(["go", "test", "./..."])
        elif package_json.exists():
            if package_json_has_test:
                candidates.append(["npm", "test", "--", "--silent"])
            candidates.append(["pnpm", "test", "--", "--silent"])
            candidates.append(["yarn", "test", "--silent"])

        if makefile_has_test:
            candidates.append(["make", "test"])

        # Common runners as a last resort.
        candidates.extend(
            [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]
        )

        seen: set[tuple[str, ...]] = set()
        unique: list[list[str]] = []
        for cmd in candidates:
            key = tuple(cmd)
            if key not in seen:
                seen.add(key)
                unique.append(cmd)
        return unique

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("running_tests", goal=goal, num_tasks=len(tasks))
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
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = self._detect_test_commands()
        logger.info("detected_test_runners", candidates=candidates)
        if not candidates:
            logger.warning("no_test_runners_detected")
            return "No test runner found."

        timeout = self._timeout
        for cmd in candidates:
            logger.debug("trying_test_runner", cmd=cmd, timeout=timeout)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except FileNotFoundError:
                logger.debug("runner_not_available", runner=cmd[0])
                continue
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
                logger.info("test_runner_completed", cmd=cmd)
                return output
            except asyncio.TimeoutError:
                logger.warning("test_runner_timeout", cmd=cmd, timeout=timeout)
                proc.kill()
                continue
            except OSError as e:
                logger.warning("test_runner_os_error", cmd=cmd, error=str(e))
                continue
        logger.error("all_test_runners_failed")
        return "No test runner found."
