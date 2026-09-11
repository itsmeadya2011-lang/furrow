from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def output(self) -> str:
        parts = [f"$ {self.command}", f"exit code: {self.exit_code}"]
        if self.timed_out:
            parts.append("TIMED OUT")
        if self.stdout:
            parts.append("--- stdout ---")
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append("--- stderr ---")
            parts.append(self.stderr.rstrip())
        return "\n".join(parts)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.workspace: Path = Path(self.client.settings.workspace)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
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

    async def _run_command(self, cmd: list[str], timeout: int = 120) -> CommandResult:
        """Run a subprocess command, capturing exit code and output.

        Returns a CommandResult. If the process times out, the process is killed
        and ``timed_out`` is set to True.
        """
        command_str = " ".join(cmd)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return CommandResult(command=command_str, exit_code=-1, stdout="", stderr="command not found")
        except Exception as e:
            return CommandResult(command=command_str, exit_code=-1, stdout="", stderr=str(e))

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return CommandResult(
                command=command_str,
                exit_code=proc.returncode if proc.returncode is not None else -1,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return CommandResult(
                command=command_str,
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds.",
                timed_out=True,
            )

    def _detect_test_commands(self) -> list[list[str]]:
        """Detect test runners based on project configuration files."""
        candidates: list[list[str]] = []

        pyproject = self.workspace / "pyproject.toml"
        if pyproject.exists():
            candidates.append(["python", "-m", "pytest", "-q"])
            candidates.append(["pytest", "-q"])

        pkg_json = self.workspace / "package.json"
        if pkg_json.exists():
            try:
                import json as _json

                data = _json.loads(pkg_json.read_text())
                scripts = data.get("scripts", {})
                if "test" in scripts:
                    candidates.extend([
                        ["npm", "test", "--", "--silent"],
                        ["pnpm", "test", "--", "--silent"],
                        ["yarn", "test", "--silent"],
                    ])
            except Exception:
                pass

        if (self.workspace / "Cargo.toml").exists():
            candidates.append(["cargo", "test", "-q"])

        if (self.workspace / "go.mod").exists():
            candidates.append(["go", "test", "./..."])

        makefile = self.workspace / "Makefile"
        if makefile.exists():
            try:
                content = makefile.read_text()
                if "\ttest" in content or "test:" in content:
                    candidates.append(["make", "test"])
            except Exception:
                pass

        # Linting / type-checking tools
        if pyproject.exists() or (self.workspace / "ruff.toml").exists():
            candidates.append(["ruff", "check", "."])
        if pyproject.exists() or (self.workspace / "mypy.ini").exists() or (self.workspace / ".mypy.ini").exists():
            candidates.append(["mypy", "."])

        return candidates

    async def _run_tests(self) -> str:
        candidates = self._detect_test_commands()
        if not candidates:
            return (
                "No test runner detected. Looked for pyproject.toml (pytest), "
                "package.json (npm/pnpm/yarn test), Cargo.toml (cargo test), "
                "go.mod (go test), Makefile (make test), and linting configs "
                "(ruff, mypy)."
            )

        results: list[CommandResult] = []
        for cmd in candidates:
            result = await self._run_command(cmd)
            results.append(result)
            # If a command ran successfully (exit 0) and did not time out, we are done.
            if result.exit_code == 0 and not result.timed_out:
                return "\n\n".join(r.output for r in results)

        # All commands failed or timed out; report the combined output.
        return "\n\n".join(r.output for r in results)
