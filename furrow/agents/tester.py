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

    async def _run_tests(self) -> str:
        # Detect project type and run appropriate tests
        project_type = self._detect_project_type()
        
        if project_type == "python":
            return await self._run_python_tests()
        elif project_type == "javascript":
            return await self._run_js_tests()
        elif project_type == "rust":
            return await self._run_rust_tests()
        elif project_type == "go":
            return await self._run_go_tests()
        else:
            # Fallback: try common test commands
            return await self._run_fallback_tests()

    def _detect_project_type(self) -> str:
        cwd = Path.cwd()
        if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists() or (cwd / "requirements.txt").exists():
            return "python"
        if (cwd / "package.json").exists():
            return "javascript"
        if (cwd / "Cargo.toml").exists():
            return "rust"
        if (cwd / "go.mod").exists():
            return "go"
        return "unknown"

    async def _run_python_tests(self) -> str:
        candidates = [
            ["pytest", "-q", "--tb=short"],
            ["python", "-m", "pytest", "-q", "--tb=short"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No Python test runner found or no tests collected."

    async def _run_js_tests(self) -> str:
        # Check package.json for test script
        try:
            import json
            with open("package.json") as f:
                pkg = json.load(f)
            if "test" in pkg.get("scripts", {}):
                # Detect package manager
                if Path("pnpm-lock.yaml").exists():
                    cmd = ["pnpm", "test", "--", "--silent"]
                elif Path("yarn.lock").exists():
                    cmd = ["yarn", "test", "--silent"]
                else:
                    cmd = ["npm", "test", "--", "--silent"]
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
        except Exception:
            pass
        return "No JavaScript test script found."

    async def _run_rust_tests(self) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "cargo", "test", "--quiet",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
                return stdout.decode() + stderr.decode()
            except asyncio.TimeoutError:
                proc.kill()
        except Exception:
            pass
        return "No Rust tests found."

    async def _run_go_tests(self) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "go", "test", "./...",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                return stdout.decode() + stderr.decode()
            except asyncio.TimeoutError:
                proc.kill()
        except Exception:
            pass
        return "No Go tests found."

    async def _run_fallback_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
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
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
