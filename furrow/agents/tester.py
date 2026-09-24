from __future__ import annotations

import asyncio
import json

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or Settings())
        self.settings = self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:  # noqa: BLE001
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        lint_output = await self._run_linting()
        type_output = await self._run_type_checking()

        combined_output = f"=== Tests ===\n{test_output}\n\n=== Linting ===\n{lint_output}\n\n=== Type Checking ===\n{type_output}"

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest/Lint/Type output:\n{combined_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
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
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (asyncio.TimeoutError, FileNotFoundError):
                continue
        return "No test runner found or no tests to run."

    async def _run_linting(self) -> str:
        """Run linters for various languages."""
        workspace = self.settings.workspace
        outputs = []

        # Python: ruff
        if (workspace / "pyproject.toml").exists() or (workspace / "ruff.toml").exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ruff",
                    "check",
                    ".",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                output = stdout.decode() + stderr.decode()
                if output.strip():
                    outputs.append(f"ruff:\n{output}")
            except (asyncio.TimeoutError, FileNotFoundError):
                pass

        # JavaScript/TypeScript: eslint
        if (workspace / "package.json").exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "npx",
                    "eslint",
                    ".",
                    "--quiet",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                output = stdout.decode() + stderr.decode()
                if output.strip():
                    outputs.append(f"eslint:\n{output}")
            except (asyncio.TimeoutError, FileNotFoundError):
                pass

        # Go: golangci-lint
        if (workspace / "go.mod").exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "golangci-lint",
                    "run",
                    "./...",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                output = stdout.decode() + stderr.decode()
                if output.strip():
                    outputs.append(f"golangci-lint:\n{output}")
            except (asyncio.TimeoutError, FileNotFoundError):
                pass

        # Rust: clippy
        if (workspace / "Cargo.toml").exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "cargo",
                    "clippy",
                    "--",
                    "-D",
                    "warnings",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                output = stdout.decode() + stderr.decode()
                if output.strip():
                    outputs.append(f"clippy:\n{output}")
            except (asyncio.TimeoutError, FileNotFoundError):
                pass

        return "\n".join(outputs) if outputs else "No linting issues found or no linter configured."

    async def _run_type_checking(self) -> str:
        """Run type checkers for various languages."""
        workspace = self.settings.workspace
        outputs = []

        # Python: mypy
        if (workspace / "pyproject.toml").exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "mypy",
                    ".",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode() + stderr.decode()
                if output.strip():
                    outputs.append(f"mypy:\n{output}")
            except (asyncio.TimeoutError, FileNotFoundError):
                pass

        # TypeScript: tsc
        if (workspace / "tsconfig.json").exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "npx",
                    "tsc",
                    "--noEmit",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode() + stderr.decode()
                if output.strip():
                    outputs.append(f"tsc:\n{output}")
            except (asyncio.TimeoutError, FileNotFoundError):
                pass

        return (
            "\n".join(outputs)
            if outputs
            else "No type checking issues found or no type checker configured."
        )
