from __future__ import annotations

import asyncio
import json
from pathlib import Path
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
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            response = await asyncio.wait_for(
                self.client.complete(prompt, model=self.client.settings.tester_model),
                timeout=60,
            )
        except asyncio.TimeoutError:
            return TestResult(
                passed=False,
                summary="LLM evaluation timed out after 60 seconds",
                failures=["LLM evaluation timeout"],
            )

        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed="passed" in response.lower(), summary=response, failures=[]
            )

    async def _run_tests(self) -> str:
        workspace = Path.cwd()
        test_command = self._detect_test_command(workspace)

        if test_command is None:
            return "No test runner found. Checked for: pytest.ini, pyproject.toml, package.json, Cargo.toml, go.mod"

        try:
            proc = await asyncio.create_subprocess_exec(
                *test_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode() + stderr.decode()
                return f"Exit code: {proc.returncode}\n{output}"
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return "Test command timed out after 120 seconds"
        except FileNotFoundError:
            return f"Test command not found: {test_command[0]}"
        except Exception as e:
            return f"Test command failed: {e}"

    def _detect_test_command(self, workspace: Path) -> list[str] | None:
        if (workspace / "pytest.ini").exists() or (workspace / "pyproject.toml").exists():
            pyproject = workspace / "pyproject.toml"
            if pyproject.exists():
                content = pyproject.read_text()
                if "pytest" in content or "[tool.pytest" in content:
                    return ["pytest", "-q"]
            if (workspace / "pytest.ini").exists():
                return ["pytest", "-q"]

        if (workspace / "package.json").exists():
            package_json = workspace / "package.json"
            try:
                content = json.loads(package_json.read_text())
                if "test" in content.get("scripts", {}):
                    if (workspace / "pnpm-lock.yaml").exists():
                        return ["pnpm", "test", "--", "--silent"]
                    if (workspace / "yarn.lock").exists():
                        return ["yarn", "test", "--silent"]
                    return ["npm", "test", "--", "--silent"]
            except (json.JSONDecodeError, OSError):
                pass

        if (workspace / "Cargo.toml").exists():
            return ["cargo", "test", "-q"]

        if (workspace / "go.mod").exists():
            return ["go", "test", "./..."]

        return None