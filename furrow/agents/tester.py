from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult, settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


MAX_TEST_OUTPUT_CHARS = 12 * 1024


@dataclass
class TestExecution:
    command: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else getattr(client, "settings", None) or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        execution = await self._run_tests()
        output = self._format_execution(execution)

        if execution.error == "No test runner found.":
            return TestResult(passed=False, summary="No test runner found.")
        if execution.error:
            return TestResult(passed=False, summary=execution.error, failures=[output])
        if execution.returncode == 0 and not execution.timed_out:
            summary = f"Tests passed (return code 0): {shlex.join(execution.command)}"
            if execution.stdout or execution.stderr:
                summary += f"\n{self._bounded(output)}"
            return TestResult(passed=True, summary=summary)

        try:
            response = await self.client.complete(
                f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{output}\n",
                model=self.settings.tester_model,
            )
        except Exception as exc:
            return self.normalize_test_result(
                None,
                process_returncode=execution.returncode,
                timed_out=execution.timed_out,
                output=f"{output}\nLLM summary failed: {exc}",
            )

        parsed = self._parse_llm_result(response)
        return self.normalize_test_result(
            parsed,
            process_returncode=execution.returncode,
            timed_out=execution.timed_out,
            output=output,
        )

    @classmethod
    def normalize_test_result(
        cls,
        result: TestResult | dict[str, object] | None,
        *,
        process_returncode: int | None,
        timed_out: bool = False,
        output: str = "",
    ) -> TestResult:
        process_passed = process_returncode == 0 and not timed_out
        if process_passed:
            return TestResult(passed=True, summary="Tests passed (return code 0).", failures=[])

        summary = "Tests failed."
        failures: list[str] = []
        if isinstance(result, TestResult):
            if isinstance(result.summary, str) and result.summary.strip():
                summary = result.summary.strip()
            failures = [failure for failure in result.failures if isinstance(failure, str)]
        elif isinstance(result, dict):
            raw_summary = result.get("summary")
            raw_failures = result.get("failures")
            if isinstance(raw_summary, str) and raw_summary.strip():
                summary = raw_summary.strip()
            if isinstance(raw_failures, list):
                failures = [failure for failure in raw_failures if isinstance(failure, str)]

        if not failures:
            failures = [output] if output.strip() else [cls._default_failure(process_returncode, timed_out)]
        return TestResult(
            passed=False,
            summary=cls._bounded(summary),
            failures=[cls._bounded(failure) for failure in failures],
        )

    @staticmethod
    def _default_failure(process_returncode: int | None, timed_out: bool) -> str:
        if timed_out:
            return "Test command timed out."
        if process_returncode is None:
            return "Test command did not complete successfully."
        return f"Test command exited with return code {process_returncode}."

    @staticmethod
    def _parse_llm_result(response: str) -> TestResult | dict[str, object] | None:
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        if type(data.get("passed")) is not bool:
            return None
        if not isinstance(data.get("summary"), str):
            return None
        if not isinstance(data.get("failures"), list) or not all(
            isinstance(failure, str) for failure in data["failures"]
        ):
            return None
        return data

    async def _run_tests(self) -> TestExecution:
        command, error = self._detect_test_command()
        if error:
            return TestExecution(command=command or [], returncode=None, stdout="", stderr="", error=error)

        workspace = Path(self.settings.workspace).expanduser()
        try:
            process = await self._create_process(command, workspace)
        except Exception as exc:
            return TestExecution(
                command=command,
                returncode=None,
                stdout="",
                stderr="",
                error=str(exc),
            )

        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.settings.test_timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            self._terminate_process(process)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=1)
            except asyncio.TimeoutError:
                stdout, stderr = b"", b""
                try:
                    await process.wait()
                except Exception:
                    pass

        returncode = process.returncode if process.returncode is not None else (-signal.SIGKILL if timed_out else None)
        return TestExecution(
            command=command,
            returncode=returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            timed_out=timed_out,
        )

    @staticmethod
    async def _create_process(command: list[str], workspace: Path) -> asyncio.subprocess.Process:
        kwargs = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "cwd": workspace,
        }
        try:
            return await asyncio.create_subprocess_exec(*command, start_new_session=True, **kwargs)
        except TypeError:
            return await asyncio.create_subprocess_exec(*command, **kwargs)

    @staticmethod
    def _terminate_process(process: asyncio.subprocess.Process) -> None:
        killpg = getattr(os, "killpg", None)
        if killpg is not None and process.pid:
            try:
                killpg(process.pid, signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass

    def _detect_test_command(self) -> tuple[list[str], str | None]:
        if self.settings.test_command is not None:
            try:
                command = shlex.split(self.settings.test_command)
            except ValueError as exc:
                return [], f"Invalid test command: {exc}"
            if not command:
                return [], "Test command is empty."
            return command, None

        workspace = Path(self.settings.workspace).expanduser()
        pytest_configured = any(
            (workspace / filename).is_file()
            for filename in ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")
        )
        if pytest_configured or shutil.which("pytest"):
            return ["pytest", "-q"], None
        if (workspace / "pnpm-lock.yaml").is_file():
            return ["pnpm", "test"], None
        if (workspace / "yarn.lock").is_file():
            return ["yarn", "test"], None
        if (workspace / "package.json").is_file():
            return ["npm", "test"], None
        if (workspace / "Cargo.toml").is_file():
            return ["cargo", "test"], None
        if (workspace / "go.mod").is_file():
            return ["go", "test", "./..."], None
        return [], "No test runner found."

    def _format_execution(self, execution: TestExecution) -> str:
        command = shlex.join(execution.command) if execution.command else "(none)"
        returncode = "none" if execution.returncode is None else str(execution.returncode)
        timeout = " (timed out)" if execution.timed_out else ""
        stdout = self._bounded(execution.stdout) or "(empty)"
        stderr = self._bounded(execution.stderr) or "(empty)"
        return (
            f"Command: {command}\n"
            f"Return code: {returncode}{timeout}\n"
            f"Stdout:\n{stdout}\n"
            f"Stderr:\n{stderr}"
        )

    @staticmethod
    def _bounded(value: str, limit: int = MAX_TEST_OUTPUT_CHARS) -> str:
        if len(value) <= limit:
            return value
        return value[:limit] + "\n... [output truncated]"
