from __future__ import annotations

import asyncio
import json
from pathlib import Path

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class TesterAgent:
    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        runner_missing = False
        try:
            test_output, runner_missing = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        if runner_missing:
            # Preserve specific error messages (e.g., invalid workspace) instead of generic message
            if "Workspace is not a valid directory" in test_output:
                return TestResult(
                    passed=False,
                    summary=test_output,
                    failures=[test_output],
                )
            return TestResult(
                passed=False,
                summary="No test runner found in workspace",
                failures=["No test runner (pytest, npm, cargo, go) found in workspace"],
            )

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed=False,
                summary="Malformed tester response",
                failures=[f"Malformed tester response: {response}"],
            )

    async def _run_tests(self) -> tuple[str, bool]:
        candidates: list[tuple[list[str], str]] = [
            (["pytest", "-q"], "pytest"),
            (["python", "-m", "pytest", "-q"], "pytest"),
            (["npm", "test", "--", "--silent"], "npm"),
            (["pnpm", "test", "--", "--silent"], "pnpm"),
            (["yarn", "test", "--silent"], "yarn"),
            (["cargo", "test", "-q"], "cargo"),
            (["go", "test", "./..."], "go"),
        ]
        workspace = self.settings.workspace
        if isinstance(workspace, str):
            workspace = Path(workspace)

        if not workspace.is_dir():
            return f"Workspace is not a valid directory: {workspace}", True

        for cmd, runner_name in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
            except FileNotFoundError:
                continue
            except OSError as e:
                return f"Failed to start {runner_name}: {e}", False

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode() + stderr.decode()
                if proc.returncode != 0:
                    return output, False
                return output, False
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
                continue
            except Exception as e:
                return f"Error running {runner_name}: {e}", False

        return "No test runner found.", True