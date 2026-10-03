from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

NO_RUNNER_OUTPUT = "No test runner detected for this project."
"""Deterministic output used when no marker file justifies a test runner."""

MAX_FAILURES = 20
"""Upper bound on the failure lines reported for one run."""

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_FAILURE_LINE_RE = re.compile(
    r"^(?:FAILED|ERROR)\b"
    r"|^---\s+(?:FAIL|ERROR)\b"
    r"|^test result:\s*FAILED"
    r"|^FAIL\b"
    r"|^panic:"
    r"|^error(?:\[|:)"
    r"|^[✗✘×]"
    r"|^\s*[✗✘×]"
    r"|^\s*(?:AssertionError|ValueError|TypeError|KeyError|Exception|RuntimeError)\b"
)


@dataclass
class TestRun:
    """Raw outcome of one test-runner invocation."""

    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    output: str = ""
    found: bool = False

    @property
    def command_text(self) -> str:
        return shlex.join(self.command) if self.command else ""


class TesterAgent:
    """Runs the project's test suite and turns the result into a :class:`TestResult`."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client if client is not None else LLMClient(settings=settings)
        self.settings = settings if settings is not None else self.client.settings

    # ---------------------------------------------------------------- detection

    def detect_command(self) -> list[str] | None:
        """Return the test command justified by a marker file, else ``None``."""
        workspace = Path(self.settings.workspace)
        if self._is_python_project(workspace):
            return ["python", "-m", "pytest", "-q"]
        node_command = self._detect_node(workspace)
        if node_command:
            return node_command
        if (workspace / "Cargo.toml").is_file():
            return ["cargo", "test", "-q"]
        if (workspace / "go.mod").is_file():
            return ["go", "test", "./..."]
        return None

    @staticmethod
    def _is_python_project(workspace: Path) -> bool:
        if (workspace / "tests").is_dir():
            return True
        if any((workspace / name).is_file() for name in ("pytest.ini", "tox.ini", "setup.cfg")):
            return True
        pyproject = workspace / "pyproject.toml"
        if pyproject.is_file():
            try:
                text = pyproject.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False
            return "tool.pytest.ini_options" in text
        return False

    @staticmethod
    def _detect_node(workspace: Path) -> list[str] | None:
        manifest = workspace / "package.json"
        if not manifest.is_file():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            return None
        scripts = data.get("scripts") if isinstance(data, dict) else None
        if not isinstance(scripts, dict) or "test" not in scripts:
            return None
        return ["npm", "test", "--", "--silent"]

    # ---------------------------------------------------------------- execution

    async def run_tests(self) -> TestRun:
        """Run the configured or detected test command in ``settings.workspace``."""
        command = self._command()
        if not command:
            return TestRun(command=[], returncode=None, output=NO_RUNNER_OUTPUT, found=False)
        timeout = max(1.0, float(self.settings.test_timeout))
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.settings.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            return TestRun(
                command=command,
                returncode=None,
                output=f"Could not run {shlex.join(command)}: {error}",
                found=False,
            )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            await self._terminate(process)
            return TestRun(
                command=command,
                returncode=None,
                output=(
                    f"Test command timed out after {timeout:.0f}s and was killed: "
                    f"{shlex.join(command)}"
                ),
                found=True,
            )
        output = (stdout or b"").decode("utf-8", errors="replace") + (
            stderr or b""
        ).decode("utf-8", errors="replace")
        return TestRun(
            command=command,
            returncode=process.returncode,
            output=output,
            found=True,
        )

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        """Kill a timed-out process and reap it so nothing is left behind."""
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()
        with contextlib.suppress(ProcessLookupError, OSError, asyncio.CancelledError):
            await process.wait()

    def _command(self) -> list[str]:
        configured = (self.settings.test_command or "").strip()
        if configured:
            return shlex.split(configured)
        return self.detect_command() or []

    # ------------------------------------------------------------------ verdict

    async def run(self, goal: str, tasks: list[TaskModel], context: str = "") -> TestResult:
        """Run the suite and report a verdict; the exit code is authoritative."""
        run = await self.run_tests()
        if not run.found:
            return TestResult(
                passed=True,
                summary=f"Tests skipped: {run.output}",
                failures=[],
                command=run.command_text or None,
                returncode=run.returncode,
            )
        limit = max(200, int(self.settings.max_output_chars))
        output = self._clean_output(run.output)
        excerpt = output[-limit:] if len(output) > limit else output
        passed = run.returncode == 0
        summary = self._deterministic_summary(run, passed)
        failures = [] if passed else self._extract_failures(output)
        verdict = await self._ask_model(goal, tasks, context, run, excerpt)
        if verdict is not None:
            summary = self._string_field(verdict, "summary") or summary
            model_failures = self._failure_field(verdict)
            if model_failures:
                failures = model_failures
            elif not passed and not failures:
                failures = [f"Test command exited with code {run.returncode}"]
        return TestResult(
            passed=passed,
            summary=summary,
            failures=failures,
            command=run.command_text or None,
            returncode=run.returncode,
        )

    async def _ask_model(
        self,
        goal: str,
        tasks: list[TaskModel],
        context: str,
        run: TestRun,
        excerpt: str,
    ) -> Any:
        """Summarize the run with the model; ``None`` when the LLM is unavailable."""
        task_lines = "\n".join(
            f"- [{task.id}] {task.description} (status: {task.status})" for task in tasks
        )
        prompt = "\n".join(
            [
                TESTER_PROMPT,
                f"\nGoal: {goal}",
                f"\nTest command: {run.command_text}",
                f"Exit code: {run.returncode}",
                f"\nTasks in this cycle:\n{task_lines or '- none'}",
                f"\nProject context:\n{context}" if context.strip() else "",
                f"\nTest output (tail):\n{excerpt}",
            ]
        )
        try:
            return await self.client.complete_json(prompt, model=self.settings.tester_model)
        except Exception:
            # The tester must never fail a cycle because the LLM is unavailable;
            # the deterministic verdict from the exit code stands on its own.
            return None

    def _deterministic_summary(self, run: TestRun, passed: bool) -> str:
        if run.returncode is None:
            return f"Tests did not report an exit code: {run.command_text}"
        if passed:
            return f"Tests passed: {run.command_text} exited with code 0"
        return f"Tests failed: {run.command_text} exited with code {run.returncode}"

    @staticmethod
    def _clean_output(output: str) -> str:
        return _ANSI_RE.sub("", output)

    @staticmethod
    def _string_field(verdict: Any, key: str) -> str:
        if not isinstance(verdict, dict):
            return ""
        value = verdict.get(key)
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _failure_field(verdict: Any) -> list[str]:
        if not isinstance(verdict, dict):
            return []
        values = verdict.get("failures")
        if not isinstance(values, list):
            return []
        return [value.strip() for value in values if isinstance(value, str) and value.strip()]

    @staticmethod
    def _extract_failures(output: str) -> list[str]:
        failures: list[str] = []
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped or not _FAILURE_LINE_RE.match(stripped):
                continue
            if stripped not in failures:
                failures.append(stripped)
            if len(failures) >= MAX_FAILURES:
                break
        return failures


__all__ = ["TestRun", "TesterAgent"]