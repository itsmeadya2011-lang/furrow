from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - capture runner failure into TestResult
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = self.client.settings.workspace
        candidates = self._get_ordered_candidates(workspace)
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except Exception:  # noqa: BLE001,S112 - runner binary may be missing; continue to next runner
                continue
        return "No test runner found."

    def _get_ordered_candidates(self, workspace) -> list[list[str]]:
        has_package_json = (workspace / "package.json").exists()
        has_cargo_toml = (workspace / "Cargo.toml").exists()
        has_go_mod = (workspace / "go.mod").exists()

        node_runners = [
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
        ]
        cargo_runner = [["cargo", "test", "-q"]]
        go_runner = [["go", "test", "./..."]]
        python_runners = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
        ]

        if has_package_json:
            return node_runners + cargo_runner + go_runner + python_runners
        if has_cargo_toml:
            return cargo_runner + node_runners + go_runner + python_runners
        if has_go_mod:
            return go_runner + node_runners + cargo_runner + python_runners
        return python_runners + node_runners + cargo_runner + go_runner
