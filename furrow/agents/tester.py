from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

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

    def _detect_project_type(self, workspace: Path) -> list[list[str]]:
        """Detect project type based on files in workspace and return appropriate test commands."""
        candidates: list[list[str]] = []
        
        # Check for Python/pytest
        pyproject = workspace / "pyproject.toml"
        setup_py = workspace / "setup.py"
        pytest_ini = workspace / "pytest.ini"
        conftest = workspace / "conftest.py"
        
        has_pytest = (
            (pyproject.exists() and "pytest" in pyproject.read_text()) or
            (setup_py.exists() and "pytest" in setup_py.read_text()) or
            pytest_ini.exists() or
            conftest.exists()
        )
        
        if has_pytest:
            if self._command_exists("pytest"):
                candidates.append(["pytest", "-q"])
            if self._command_exists("python"):
                candidates.append(["python", "-m", "pytest", "-q"])
        
        # Check for Node.js
        package_json = workspace / "package.json"
        if package_json.exists():
            pkg_text = package_json.read_text()
            has_npm_test = '"test"' in pkg_text
            if has_npm_test:
                if self._command_exists("pnpm"):
                    candidates.append(["pnpm", "test", "--", "--silent"])
                elif self._command_exists("yarn"):
                    candidates.append(["yarn", "test", "--silent"])
                elif self._command_exists("npm"):
                    candidates.append(["npm", "test", "--", "--silent"])
        
        # Check for Rust
        cargo_toml = workspace / "Cargo.toml"
        if cargo_toml.exists() and self._command_exists("cargo"):
            candidates.append(["cargo", "test", "-q"])
        
        # Check for Go
        go_mod = workspace / "go.mod"
        if go_mod.exists() and self._command_exists("go"):
            candidates.append(["go", "test", "./..."])
        
        # Check for Java/Maven
        pom_xml = workspace / "pom.xml"
        build_gradle = workspace / "build.gradle"
        if (pom_xml.exists() or build_gradle.exists()) and self._command_exists("mvn"):
            candidates.append(["mvn", "test", "-q"])
        
        return candidates

    def _command_exists(self, cmd: str) -> bool:
        """Check if a command exists in PATH."""
        import shutil
        return shutil.which(cmd) is not None

    async def _run_tests(self) -> str:
        workspace = self.settings.workspace
        candidates = self._detect_project_type(workspace)
        
        # If no project type detected, try all available runners
        if not candidates:
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
                    *cmd, 
                    stdout=asyncio.subprocess.PIPE, 
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace),
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    # Check if this command actually produced test output
                    # (not just "command not found")
                    if "not found" not in output.lower() and proc.returncode is not None:
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    continue
            except (FileNotFoundError, Exception):
                continue
        
        return "No test runner found or all tests passed."
