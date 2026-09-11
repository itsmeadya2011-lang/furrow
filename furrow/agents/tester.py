from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from furrow.config import TestResult

if TYPE_CHECKING:
    from furrow.config import Settings, TaskModel


class TesterAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        if self.settings is None:
            from furrow.config import settings as default_settings

            self.settings = default_settings

        command = list(self.settings.test_command)
        if not command:
            return TestResult(
                passed=False,
                summary="No test command configured (settings.test_command is empty).",
                failures=["settings.test_command is not set"],
            )

        timeout = self.settings.test_timeout_seconds
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.settings.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return TestResult(
                passed=False,
                summary=f"Test executable not found for command: {command}",
                failures=[f"Executable not found: {command[0]}"],
                command=command,
                exit_code=None,
            )
        except PermissionError:
            return TestResult(
                passed=False,
                summary=f"Permission denied running test command: {command}",
                failures=[f"Permission denied: {command[0]}"],
                command=command,
                exit_code=None,
            )
        except OSError as exc:
            return TestResult(
                passed=False,
                summary=f"Failed to start test command: {command}: {exc}",
                failures=[f"OSError: {exc}"],
                command=command,
                exit_code=None,
            )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return TestResult(
                passed=False,
                summary=f"Test command timed out after {timeout}s: {command}",
                failures=[f"Timeout after {timeout}s: {command}"],
                command=command,
                exit_code=None,
            )

        exit_code = proc.returncode if proc.returncode is not None else -1
        stdout_text = stdout.decode(errors="replace") if stdout else ""
        stderr_text = stderr.decode(errors="replace") if stderr else ""
        output = (stdout_text + stderr_text).strip()
        summary = output or f"Command exited with code {exit_code}"

        if exit_code == 0:
            return TestResult(
                passed=True,
                summary=summary,
                failures=[],
                command=command,
                exit_code=exit_code,
            )

        failure_markers = ("failed", "failure", "error", "traceback", "assertion", "assert")
        matched: list[str] = []
        seen: set[str] = set()
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if any(marker in lower for marker in failure_markers):
                if stripped not in seen:
                    seen.add(stripped)
                    matched.append(stripped)
        failures = matched[:20] if matched else [summary]
        return TestResult(
            passed=False,
            summary=summary,
            failures=failures,
            command=command,
            exit_code=exit_code,
        )