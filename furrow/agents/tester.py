from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT, extract_json
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 -- runner failures become a failed TestResult
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(extract_json(response))
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _candidate_runners(self) -> list[list[str]]:
        """Order test-runner commands by what the project appears to use."""
        candidates: list[list[str]] = []
        cwd = Path.cwd()
        has_python = (
            (cwd / "pyproject.toml").exists()
            or (cwd / "pytest.ini").exists()
            or (cwd / "tox.ini").exists()
            or (cwd / "setup.py").exists()
        )
        has_node = (cwd / "package.json").exists()
        has_cargo = (cwd / "Cargo.toml").exists()
        has_go = (cwd / "go.mod").exists()
        if has_python:
            candidates.extend([["pytest", "-q"], ["python", "-m", "pytest", "-q"]])
        if has_node:
            candidates.extend(
                [
                    ["npm", "test", "--", "--silent"],
                    ["pnpm", "test", "--", "--silent"],
                    ["yarn", "test", "--silent"],
                ]
            )
        if has_cargo:
            candidates.append(["cargo", "test", "-q"])
        if has_go:
            candidates.append(["go", "test", "./..."])
        # Fallbacks when no project markers are found.
        if not candidates:
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]
        return candidates

    async def _run_tests(self) -> str:
        for cmd in self._candidate_runners():
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except Exception as e:  # noqa: BLE001 -- try the next candidate runner
                log.debug("test_runner_unavailable", cmd=" ".join(cmd), error=str(e))
                continue
        return "No test runner found."
