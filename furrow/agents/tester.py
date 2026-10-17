from __future__ import annotations

import asyncio
import json
from pathlib import Path

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient


class TesterAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings=None,
    ) -> None:
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
        workspace = Path(self.client.settings.workspace).resolve()
        parts: list[str] = []

        # 1) Test suite: prefer pytest, run from the workspace.
        pytest_output = await self._run_pytest(workspace)
        if pytest_output:
            parts.append(f"--- pytest ---\n{pytest_output}")

        # 2) Lint: ruff if available.
        lint_output = await self._run_ruff(workspace)
        if lint_output:
            parts.append(f"--- ruff ---\n{lint_output}")

        if not parts:
            return "No test runner or linter found."
        return "\n\n".join(parts)

    async def _run_pytest(self, workspace: Path) -> str:
        for cmd in (["pytest", "-q"], ["python", "-m", "pytest", "-q"]):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                return stdout.decode() + stderr.decode()
            except (FileNotFoundError, asyncio.TimeoutError, Exception):
                continue
        return ""

    async def _run_ruff(self, workspace: Path) -> str:
        for cmd in (["ruff", "check"], ["python", "-m", "ruff", "check"]):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode() + stderr.decode()
                return output if output.strip() else "ruff check: no issues"
            except (FileNotFoundError, asyncio.TimeoutError, Exception):
                continue
        return ""
