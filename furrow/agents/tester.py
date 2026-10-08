from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = await self._run_tests()

        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Completed tasks: {[t.id for t in tasks if t.status == 'completed']}\n\n"
            f"Test output:\n{test_output}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            lowered = response.lower()
            passed = "passed" in lowered and "fail" not in lowered
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        cwd = os.getcwd()
        outputs: list[str] = []

        # Prefer the project's own test runner when one is discoverable.
        runner = self._detect_runner(cwd)
        if runner:
            outputs.append(f"$ {' '.join(runner)}\n")
            outputs.append(await self._run_command(runner, cwd=cwd))
        else:
            outputs.append("No project-specific test runner detected; falling back to pytest.\n")
            outputs.append(await self._run_pytest(cwd))

        # Also run a syntax/import check on the changed files.
        changed = self._changed_python_files(cwd)
        if changed:
            outputs.append("\n# Import/syntax checks on changed files:\n")
            for path in changed:
                outputs.append(f"$ python -m py_compile {path}\n")
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "py_compile", path,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    outputs.append(f"  OK {path}\n")
                else:
                    outputs.append(f"  FAIL {path}\n{stderr.decode()}\n")

        return "".join(outputs)

    def _detect_runner(self, cwd: str) -> list[str] | None:
        # pytest
        if os.path.exists(os.path.join(cwd, "pytest.ini")) or os.path.exists(
            os.path.join(cwd, "pyproject.toml")
        ):
            return ["python", "-m", "pytest", "-q"]
        if shutil.which("pytest"):
            return ["pytest", "-q"]
        if os.path.exists(os.path.join(cwd, "package.json")):
            return ["npm", "test", "--", "--silent"]
        if os.path.exists(os.path.join(cwd, "Cargo.toml")):
            return ["cargo", "test", "-q"]
        if os.path.exists(os.path.join(cwd, "go.mod")):
            return ["go", "test", "./..."]
        return None

    async def _run_pytest(self, cwd: str) -> str:
        return await self._run_command(["python", "-m", "pytest", "-q"], cwd=cwd)

    async def _run_command(self, cmd: list[str], cwd: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"$ {' '.join(cmd)}\n[timeout after 120s]\n"
            out = stdout.decode() + stderr.decode()
            return f"$ {' '.join(cmd)}\n{out}"
        except FileNotFoundError:
            return f"$ {' '.join(cmd)}\n[command not found]\n"
        except Exception as e:  # noqa: BLE001
            return f"$ {' '.join(cmd)}\n[error: {e}]\n"

    def _changed_python_files(self, cwd: str) -> list[str]:
        # Best-effort: collect .py files under the workspace, excluding caches.
        root = os.path.abspath(cwd)
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "node_modules", ".venv", "venv"}]
            for name in filenames:
                if name.endswith(".py"):
                    files.append(os.path.join(dirpath, name))
        return files