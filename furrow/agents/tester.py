from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_TEST_RUNNERS: list[tuple[list[str], list[str]]] = [
    (["pytest", "-q"], ["pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py"]),
    (["python", "-m", "pytest", "-q"], ["pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py"]),
    (["npm", "test", "--", "--silent"], ["package.json", "package-lock.json"]),
    (["pnpm", "test", "--", "--silent"], ["package.json", "pnpm-lock.yaml"]),
    (["yarn", "test", "--silent"], ["package.json", "yarn.lock"]),
    (["cargo", "test", "-q"], ["Cargo.toml"]),
    (["go", "test", "./..."], ["go.mod"]),
]


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        """Run the project's test suite using the first matching test runner.

        Prefers runners whose lock/manifest files exist in the workspace.
        """
        workspace = str(self.client.settings.workspace)
        candidates = self._detect_test_runners(workspace)

        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
        return "No test runner found."

    def _detect_test_runners(self, workspace: str) -> list[list[str]]:
        """Return candidate commands ordered by preference, filtered by manifest presence."""
        manifests = set()
        try:
            for entry in os.listdir(workspace):
                manifests.add(entry)
        except OSError:
            pass

        ordered: list[list[str]] = []
        for cmd, markers in _TEST_RUNNERS:
            if any(m in manifests for m in markers):
                ordered.append(cmd)
        # Append any remaining runners not matched by manifest
        seen = {tuple(c) for c in ordered}
        for cmd, _ in _TEST_RUNNERS:
            if tuple(cmd) not in seen:
                ordered.append(cmd)
        return ordered
