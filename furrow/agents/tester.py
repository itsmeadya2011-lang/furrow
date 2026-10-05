from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

from furrow.agents.base import BaseAgent, extract_json
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult

TEST_TIMEOUT_SECONDS = 120
# 0 = the runner succeeded, 1 = tests ran and some failed.  Anything else (pytest 4 =
# usage error, 5 = nothing collected, npm = bad script, ...) means this runner cannot
# say anything about this project, so we try the next candidate instead of reporting it
# as a test failure.
USABLE_EXIT_CODES = (0, 1)
MAX_TASK_RESULT_CHARS = 500


def _module_available(module: str) -> bool:
    """True when ``module`` is importable by the interpreter running furrow."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _decode(stream: bytes | None) -> str:
    """Decode a captured subprocess stream, tolerating a missing or non-UTF-8 payload."""
    return (stream or b"").decode(errors="replace")


def _test_runners() -> list[tuple[str, ...]]:
    """Candidate test runners in priority order.

    The pytest module fallback uses the running interpreter instead of a bare
    ``python`` and is only offered when pytest is actually importable: a missing
    pytest exits 1, which would otherwise be mistaken for "tests ran and failed".
    """
    runners: list[tuple[str, ...]] = [("pytest", "-q")]
    if _module_available("pytest"):
        runners.append((sys.executable, "-m", "pytest", "-q"))
    runners.extend(
        [
            ("npm", "test", "--", "--silent"),
            ("pnpm", "test", "--", "--silent"),
            ("yarn", "test", "--silent"),
            ("cargo", "test", "-q"),
            ("go", "test", "./..."),
        ]
    )
    return runners


class TesterAgent(BaseAgent):
    """Runs the project's tests and turns the output into a validated verdict."""

    # Not a pytest test class: pytest collects any imported name starting with
    # "Test", and this one has an ``__init__``. Opt out explicitly so importing
    # it from a test module does not raise a collection warning.
    __test__ = False

    model_attr = "tester_model"

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            exit_code, test_output = await self._run_tests()
        except (OSError, asyncio.TimeoutError, UnicodeDecodeError) as exc:
            message = f"Could not run the test suite: {exc}"
            return TestResult(passed=False, summary=message, failures=[message])

        prompt = (
            f"Goal: {goal}\n\n"
            f"Test runner exit code: {self._describe_exit_code(exit_code)}\n\n"
            f"Worker reports:\n{self._task_digest(tasks)}\n\n"
            f"Test output:\n{test_output}\n"
        )
        response = await self.ask(prompt, system=TESTER_PROMPT)
        return self._verdict(response)

    def _verdict(self, response: str) -> TestResult:
        """Validate the tester response, failing closed when it is not usable JSON.

        The verdict is never inferred from free text: an unparseable response is a
        failure, because guessing is how a broken build gets reported as green.
        """
        try:
            return TestResult.model_validate(json.loads(extract_json(response)))
        except (json.JSONDecodeError, ValueError):
            return TestResult(
                passed=False,
                summary="Could not parse tester JSON",
                failures=[response],
            )

    async def _run_tests(self) -> tuple[int | None, str]:
        """Run the first usable test runner in the configured workspace.

        Returns the runner's exit code (``None`` when no runner could be used) plus its
        output, with diagnostics appended when nothing ran.
        """
        workspace = self._effective_settings.workspace
        if not await asyncio.to_thread(Path(workspace).is_dir):
            return None, f"Workspace directory does not exist: {workspace}"

        diagnostics: list[str] = []
        for cmd in _test_runners():
            label = " ".join(cmd)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
                diagnostics.append(f"{label}: unusable ({exc})")
                continue

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=TEST_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                await self._reap(proc)
                diagnostics.append(f"{label}: timed out after {TEST_TIMEOUT_SECONDS}s")
                continue

            if proc.returncode in USABLE_EXIT_CODES:
                output = _decode(stdout) + _decode(stderr)
                return proc.returncode, output
            diagnostics.append(f"{label}: unusable exit code {proc.returncode}")

        return None, self._no_runner_report(diagnostics)

    @staticmethod
    async def _reap(proc: asyncio.subprocess.Process) -> None:
        """Kill a timed-out child and wait for it, so no zombie is left behind."""
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()

    @staticmethod
    def _no_runner_report(diagnostics: list[str]) -> str:
        report = "No usable test runner found in the workspace."
        if not diagnostics:
            return report
        details = "\n".join(f"- {entry}" for entry in diagnostics)
        return f"{report}\nDiagnostics:\n{details}"

    @staticmethod
    def _describe_exit_code(exit_code: int | None) -> str:
        if exit_code is None:
            return "unavailable - no test runner could be executed"
        if exit_code == 0:
            return "0 - the test runner completed successfully"
        if exit_code == 1:
            return "1 - tests ran and at least one failed"
        return f"{exit_code} - the test runner could not run the suite"

    @staticmethod
    def _task_digest(tasks: list[TaskModel]) -> str:
        """Per-task context (ids, dependencies, status and results) for the tester."""
        if not tasks:
            return "(no tasks reported)"

        lines: list[str] = []
        for task in tasks:
            depends = ", ".join(task.dependencies) if task.dependencies else "none"
            result = (task.result or "").strip() or "(no result reported)"
            if len(result) > MAX_TASK_RESULT_CHARS:
                result = result[:MAX_TASK_RESULT_CHARS] + "... [truncated]"
            lines.append(f"- id={task.id} status={task.status} depends_on={depends}")
            lines.append(f"  description: {task.description}")
            lines.append(f"  result: {result}")
        return "\n".join(lines)
