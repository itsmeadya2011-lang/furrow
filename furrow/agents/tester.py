"""Detect a test runner, run it, and turn its real output into a :class:`TestResult`.

Two properties matter here.  First, the verdict comes from the *actual* exit code, not
from a model's opinion about it: a green suite is reported without spending a single
token, and a red one is always red even if the model says otherwise.  Second, nothing
in this module raises -- a missing runner, a vanished binary, a hung suite or an
unparseable model reply all degrade into a :class:`TestResult`, because verification
must never be the thing that breaks a run.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.util import extract_json_object, truncate

if TYPE_CHECKING:
    from furrow.config import Settings

__all__ = ["TesterAgent"]

_PYTHON_MARKERS = (
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
    "setup.py",
    "conftest.py",
)
_PYTHON_DIRS = ("tests", "test")
_DEFAULT_TIMEOUT = 180.0
_DEFAULT_MAX_OUTPUT = 20000
_SUMMARY_LINES = 20
_MAX_FAILURES = 20
_GOAL_CHARS = 2000
_ONE_LINER = 300
_FAILURE_RE = re.compile(r"FAILED|ERROR|Error:|assert|\b\d+\s+(?:failed|error)")
_TIMEOUT_CODE = 124
_MISSING_BINARY_CODE = 127
_KILL_GRACE = 5.0
_NO_RUNNER = "No test runner detected - verification skipped"


class TesterAgent:
    """Runs the workspace's test command and summarizes the outcome."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client if client is not None else LLMClient(settings=settings)
        self.settings: Settings = settings if settings is not None else self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the detected test command and report what happened.

        Args:
            goal: The run's goal, given to the model so it can judge intent.
            tasks: The tasks that were planned or executed this cycle.

        Returns:
            A :class:`TestResult` that always carries the command and exit code.  When
            no runner exists the result is ``passed=True`` with ``command=None``: a
            workspace without tests is not a failing workspace, and treating "no
            evidence" as "failure" would send the orchestrator into pointless repair
            cycles.  A suite that cannot even start (``exit 127``) *is* reported as a
            failure, because that is a broken workspace, not an untested one.
        """
        cmd = await self.detect_command()
        if cmd is None:
            return TestResult(
                passed=True, summary=_NO_RUNNER, failures=[], exit_code=None, command=None
            )

        code, output = await self.run_command(cmd)
        command = " ".join(cmd)
        if code == 0:
            # A green suite needs no interpretation, so no model call is made.
            return TestResult(
                passed=True,
                summary=self._pass_summary(command, output),
                failures=[],
                exit_code=code,
                command=command,
            )
        return await self._diagnose(goal, tasks, code, output, command)

    async def detect_command(self) -> list[str] | None:
        """Return the first test command this workspace can actually run, else ``None``.

        Candidates keep the historical preference order (pytest, node, cargo, go) but
        each one is now gated on both the binary being on ``PATH`` and a project marker
        being present, so a runner is never invoked in a repository that has no tests
        to run.
        """
        root = self._root()
        if self._is_python_project(root):
            if shutil.which("pytest"):
                return ["pytest", "-q"]
            interpreter = shutil.which("python") or shutil.which("python3")
            if interpreter:
                return [interpreter, "-m", "pytest", "-q"]
        if (root / "package.json").is_file():
            for runner, args in (
                ("npm", ["test", "--", "--silent"]),
                ("pnpm", ["test", "--", "--silent"]),
                ("yarn", ["test", "--silent"]),
            ):
                if shutil.which(runner):
                    return [runner, *args]
        if (root / "Cargo.toml").is_file() and shutil.which("cargo"):
            return ["cargo", "test", "-q"]
        if (root / "go.mod").is_file() and shutil.which("go"):
            return ["go", "test", "./..."]
        return None

    async def run_command(self, cmd: list[str]) -> tuple[int, str]:
        """Run ``cmd`` in the workspace, returning its exit code and combined output.

        The output is decoded leniently and truncated to ``max_output_chars`` so binary
        noise or a runaway log cannot blow up the prompt.  A suite that hangs is killed
        and reported as exit ``124``; a binary that vanished between detection and
        execution is reported as exit ``127``.  Cancellation propagates.
        """
        timeout = self._timeout()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._root()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return (_MISSING_BINARY_CODE, f"could not run {' '.join(cmd)}: {exc}")

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            await self._kill(proc)
            return (_TIMEOUT_CODE, f"test command timed out after {timeout:g}s")

        combined = _decode(stdout) + _decode(stderr)
        code = proc.returncode if proc.returncode is not None else -1
        return (code, truncate(combined, limit=self._max_output()))

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process) -> None:
        """Kill a hung test process and reap it, draining its pipes so no transport leaks."""
        try:
            proc.kill()
        except OSError:
            pass
        await proc.wait()
        try:
            await asyncio.wait_for(proc.communicate(), timeout=_KILL_GRACE)
        except (asyncio.TimeoutError, TimeoutError, ValueError, OSError):
            pass

    async def _diagnose(
        self, goal: str, tasks: list[TaskModel], code: int, output: str, command: str
    ) -> TestResult:
        """Summarize a failure, falling back to a deterministic read of the output."""
        prompt = "\n\n".join(
            [
                f"Goal:\n{truncate(goal, limit=_GOAL_CHARS)}",
                f"Tasks attempted this cycle:\n{_task_lines(tasks) or '(none)'}",
                f"Test command: {command}",
                f"Exit code: {code}",
                f"Test output:\n{output or '(no output)'}",
            ]
        )
        try:
            data = extract_json_object(
                await self.client.complete(
                    prompt,
                    system=TESTER_PROMPT,
                    model=getattr(self.client.settings, "tester_model", None),
                )
            )
            summary = _as_text(data.get("summary")).strip()
            failures = _as_str_list(data.get("failures"))
            if not summary and not failures:
                raise ValueError("model returned neither a summary nor any failures")
        except Exception:
            # A missing key, a rate limit or prose instead of JSON: the raw output is
            # still good evidence, so read it directly rather than failing the run.
            return self._heuristic(code, output, command)
        # The exit code is the verdict; only the wording comes from the model.
        return TestResult(
            passed=False,
            summary=summary or f"{command} exited with code {code}",
            failures=failures,
            exit_code=code,
            command=command,
        )

    def _heuristic(self, code: int, output: str, command: str) -> TestResult:
        """Build a failure result from the output alone, with no model involved."""
        lines = [line.rstrip() for line in output.splitlines() if line.strip()]
        summary = truncate("\n".join(lines[-_SUMMARY_LINES:]), limit=self._max_output())
        if not summary:
            summary = f"{command} exited with code {code} and produced no output"
        failures: list[str] = []
        for line in lines:
            stripped = line.strip()
            if _FAILURE_RE.search(stripped) and stripped not in failures:
                failures.append(stripped)
            if len(failures) >= _MAX_FAILURES:
                break
        return TestResult(
            passed=False,
            summary=summary,
            failures=failures,
            exit_code=code,
            command=command,
        )

    @staticmethod
    def _pass_summary(command: str, output: str) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        detail = (
            truncate(lines[-1], limit=_ONE_LINER, marker=" ...")
            if lines
            else f"{command} exited 0"
        )
        return truncate(f"Tests passed: {detail}", limit=_ONE_LINER, marker=" ...")

    @staticmethod
    def _is_python_project(root: Path) -> bool:
        if any((root / marker).is_file() for marker in _PYTHON_MARKERS):
            return True
        return any((root / name).is_dir() for name in _PYTHON_DIRS)

    def _root(self) -> Path:
        return Path(getattr(self.settings, "workspace", "."))

    def _timeout(self) -> float:
        raw = getattr(self.settings, "test_command_timeout", _DEFAULT_TIMEOUT)
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return _DEFAULT_TIMEOUT

    def _max_output(self) -> int:
        raw = getattr(self.settings, "max_output_chars", _DEFAULT_MAX_OUTPUT)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return _DEFAULT_MAX_OUTPUT


def _decode(raw: bytes | None) -> str:
    return (raw or b"").decode("utf-8", errors="replace")


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _task_lines(tasks: list[TaskModel]) -> str:
    lines = []
    for task in tasks:
        files = f" (files: {', '.join(task.files)})" if task.files else ""
        lines.append(f"- {task.id}: {task.description}{files}")
    return "\n".join(lines)
