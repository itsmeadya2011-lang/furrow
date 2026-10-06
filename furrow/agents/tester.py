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
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _detect_test_command(self, workspace: str) -> list[str] | None:
        workspace_path = Path(workspace)

        pyproject = workspace_path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="ignore")
                if "[tool.pytest]" in content or "pytest" in content:
                    return ["pytest", "-q"]
            except OSError:
                pass

        if (workspace_path / "pytest.ini").exists() or (workspace_path / "conftest.py").exists():
            return ["pytest", "-q"]

        package_json = workspace_path / "package.json"
        if package_json.exists():
            try:
                content = package_json.read_text(encoding="utf-8", errors="ignore")
                if "test" in content:
                    return ["npm", "test", "--", "--silent"]
            except OSError:
                pass

        cargo_toml = workspace_path / "Cargo.toml"
        if cargo_toml.exists():
            return ["cargo", "test", "-q"]

        go_mod = workspace_path / "go.mod"
        if go_mod.exists():
            return ["go", "test", "./..."]

        return None

    async def _run_tests(self) -> str:
        workspace = self.client.settings.workspace
        cmd = self._detect_test_command(workspace)

        if cmd is None:
            return "No test runner found."

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode() + stderr.decode()
                return f"Command: {' '.join(cmd)}\nOutput:\n{output}"
            except asyncio.TimeoutError:
                proc.kill()
                return f"Command: {' '.join(cmd)}\nOutput: Timed out after 120 seconds"
        except FileNotFoundError:
            return f"Command: {' '.join(cmd)}\nOutput: Test runner not found ({cmd[0]})"
        except Exception as e:
            return f"Command: {' '.join(cmd)}\nOutput: Error: {e}"
