from __future__ import annotations

import asyncio
import json

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = self.settings.workspace
        if not workspace.exists():
            return "Workspace does not exist; no tests to run."

        # Prefer pytest in the workspace, then fall back to other runners.
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["python3", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
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
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode == 0:
                        return output + "\nExit code: 0 (tests passed)"
                    return output + f"\nExit code: {proc.returncode} (tests failed)"
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):  # noqa: BLE001, S112
                continue
        return "No test runner found."
