from __future__ import annotations

import asyncio
import json
import os
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
        # Collect all files that were modified in this cycle
        changed_files = []
        for task in tasks:
            changed_files.extend(task.files)

        test_output = ""
        try:
            test_output = await self._run_tests(changed_files)
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nChanged files: {', '.join(changed_files) if changed_files else 'none'}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self, changed_files: list[str]) -> str:
        # Determine project type and run appropriate tests
        workspace = Path(self.client.settings.workspace).resolve()

        # Check for test files related to changed files
        test_targets = self._get_test_targets(changed_files, workspace)

        candidates = self._get_test_commands(test_targets, workspace)

        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=workspace
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

    def _get_test_targets(self, changed_files: list[str], workspace: Path) -> list[str]:
        """Find test files related to changed files."""
        targets = []
        for f in changed_files:
            path = workspace / f
            if not path.exists():
                continue

            # Look for corresponding test files
            stem = path.stem
            parent = path.parent

            # Common test patterns
            test_patterns = [
                f"test_{stem}.py",
                f"{stem}_test.py",
                f"tests/test_{stem}.py",
                f"tests/{stem}_test.py",
                f"test/{stem}.test.ts",
                f"test/{stem}.spec.ts",
                f"{stem}.test.ts",
                f"{stem}.spec.ts",
            ]

            for pattern in test_patterns:
                test_path = workspace / pattern
                if test_path.exists():
                    targets.append(str(test_path.relative_to(workspace)))

        # Also check for general test directories
        for test_dir in ["tests", "test", "__tests__", "spec"]:
            test_path = workspace / test_dir
            if test_path.exists() and test_path.is_dir():
                targets.append(test_dir)
                break

        return targets

    def _get_test_commands(self, test_targets: list[str], workspace: Path) -> list[list[str]]:
        """Generate test commands based on project type and targets."""
        commands = []

        # Python/pytest
        if (workspace / "pyproject.toml").exists() or (workspace / "setup.py").exists() or (workspace / "requirements.txt").exists():
            if test_targets:
                for target in test_targets:
                    commands.append(["pytest", target, "-q"])
            commands.append(["pytest", "-q"])
            commands.append(["python", "-m", "pytest", "-q"])

        # JavaScript/TypeScript
        if (workspace / "package.json").exists():
            if test_targets:
                for target in test_targets:
                    commands.append(["npm", "test", "--", target, "--silent"])
            commands.append(["npm", "test", "--", "--silent"])
            commands.append(["pnpm", "test", "--", "--silent"])
            commands.append(["yarn", "test", "--silent"])

        # Rust
        if (workspace / "Cargo.toml").exists():
            commands.append(["cargo", "test", "-q"])

        # Go
        if (workspace / "go.mod").exists():
            if test_targets:
                for target in test_targets:
                    commands.append(["go", "test", "-v", target])
            commands.append(["go", "test", "./..."])

        return commands