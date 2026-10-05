"""Tester agent: run the project's real test suite and judge the verdict.

The runner is chosen from the files actually present in the workspace rather
than by trying every known test command, and the subprocess layer deliberately
avoids blanket exception handling so genuine bugs surface instead of being
reported as "no test runner found".
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import structlog

from furrow.agents.errors import AgentError
from furrow.agents.jsonutil import extract_json
from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM
from furrow.config import TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings
    from furrow.config import TaskModel

__all__ = ["TesterAgent"]

log = structlog.get_logger(__name__)

#: Default when ``Settings.test_timeout`` is absent.
_DEFAULT_TEST_TIMEOUT = 120.0
#: Maximum characters of combined test output handed to the model (tail kept).
_MAX_OUTPUT_CHARS = 8000
#: Marker prefixed to output that had to be shortened.
_TRUNCATED = "\n... [truncated] "
#: Characters of output echoed into a fallback verdict summary.
_SUMMARY_TAIL = 400
_PYTEST = ["python", "-m", "pytest", "-q"]
_FALLBACK = [_PYTEST]


class _TestRun(NamedTuple):
    """Outcome of a single test-runner attempt."""

    output: str
    returncode: int | None = None
    error: str | None = None


class TesterAgent:
    """Run the suite for ``goal`` and decide whether the work is acceptable."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        """Store the client, constructing a default one when none is supplied.

        Args:
            client: Shared :class:`~furrow.llm.LLMClient`; created if omitted.
            settings: Explicit settings; defaults to the client's own settings.
        """
        if client is None:
            self.client = LLMClient(settings=settings) if settings is not None else LLMClient()
        else:
            self.client = client
        self.settings = settings if settings is not None else getattr(self.client, "settings", None)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the tests, ask the model to judge them, and force the verdict.

        A task reported as ``failed`` fails the whole result regardless of the
        model's opinion.

        Args:
            goal: The objective the work was supposed to achieve.
            tasks: The plan tasks, used both for prompt context and as an
                automatic failure signal.

        Returns:
            A :class:`~furrow.config.TestResult`; this never raises because of
            the test runner, its output, or an unparseable verdict.
        """
        failed_ids = _failed_task_ids(tasks)
        run = await self._run_tests()
        if run.error:
            return TestResult(
                passed=False,
                summary=f"Could not run the test suite: {run.error}",
                failures=[run.error, *_failed_notes(failed_ids)],
            )

        prompt = self._build_prompt(goal, tasks, failed_ids, run)
        response = await self.client.complete(
            prompt,
            system=TESTER_SYSTEM,
            model=self._setting("tester_model", None),
        )
        result = self._verdict(response, run.output)

        if failed_ids:
            result.passed = False
            result.failures = [*_failed_notes(failed_ids), *result.failures]
        return result

    async def _run_tests(self) -> _TestRun:
        """Run the first available test command for this project type."""
        workspace = self._workspace()
        commands = detect_test_commands(workspace) or [list(_FALLBACK)]
        timeout = float(self._setting("test_timeout", _DEFAULT_TEST_TIMEOUT))
        unavailable: list[str] = []

        for command in commands:
            label = " ".join(command)
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace),
                )
            except FileNotFoundError:
                log.warning("test_runner_missing", command=label)
                unavailable.append(label)
                continue
            except OSError as exc:
                log.warning("test_runner_spawn_failed", command=label, error=str(exc)[:200])
                unavailable.append(f"{label} ({exc})")
                continue

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                message = f"Test run timed out after {timeout:g}s: {label}"
                log.warning("test_run_timeout", command=label, timeout=timeout)
                return _TestRun(output=message, returncode=None, error=message)

            combined = stdout.decode("utf-8", "replace") + stderr.decode("utf-8", "replace")
            log.info("test_run_finished", command=label, returncode=process.returncode)
            return _TestRun(output=_truncate(combined), returncode=process.returncode)

        detail = f"Tried: {', '.join(unavailable)}." if unavailable else "No candidate commands."
        message = f"No usable test runner for this project. {detail}"
        log.warning("test_runner_unavailable", tried=unavailable)
        return _TestRun(output=message, returncode=None, error=message)

    def _verdict(self, response: str, output: str) -> TestResult:
        """Turn the model's reply into a :class:`TestResult`, never assuming pass."""
        try:
            data = extract_json(response)
        except AgentError as exc:
            log.warning("tester_verdict_unparseable", error=str(exc)[:300])
            return _heuristic_verdict(response, output)

        if not isinstance(data, dict):
            return _heuristic_verdict(response, output)

        failures = data.get("failures")
        if not isinstance(failures, list):
            failures = []
        summary = data.get("summary")
        if not isinstance(summary, str):
            summary = (output or response).strip()[:_MAX_OUTPUT_CHARS]
        return TestResult(
            passed=bool(data.get("passed")),
            summary=summary,
            failures=[str(item) for item in failures],
        )

    def _build_prompt(
        self,
        goal: str,
        tasks: list[TaskModel],
        failed_ids: list[str],
        run: _TestRun,
    ) -> str:
        """Assemble goal, task status and (truncated) test output for the model."""
        lines = [f"\n\nGoal: {goal}", "\nTasks:"]
        if not tasks:
            lines.append("  (no tasks)")
        for task in tasks:
            status = getattr(task, "status", "unknown")
            lines.append(f"  [{task.id}] {status}: {_one_line(task.description)}")
        if failed_ids:
            lines.append(
                "\nAt least one task is marked failed (" + ", ".join(failed_ids) + "), so the "
                'verdict "passed" must be false.'
            )
        exit_code = "unknown" if run.returncode is None else str(run.returncode)
        lines.append(f"\nTest command exit code: {exit_code}")
        lines.append(f"\nTest output (tail):\n{run.output}\n")
        return TESTER_PROMPT + "".join(lines)

    def _workspace(self) -> Path:
        """Return the configured workspace as a :class:`~pathlib.Path`."""
        return Path(self._setting("workspace", Path.cwd()))

    def _setting(self, name: str, default: Any) -> Any:
        """Read a setting from the explicit settings or the client's settings."""
        holder = self.settings if self.settings is not None else getattr(
            self.client, "settings", None
        )
        value = getattr(holder, name, default)
        return default if value is None else value


def detect_test_commands(workspace: Path) -> list[list[str]]:
    """Pick the test commands that suit the project type in ``workspace``.

    Args:
        workspace: Project root to inspect.

    Returns:
        A list of argv lists, most specific ecosystem first. Empty when no known
        build manifest is present.
    """
    root = Path(workspace)
    commands: list[list[str]] = []

    def present(*names: str) -> bool:
        """Report whether any of ``names`` exists directly under ``root``."""
        return any((root / name).exists() for name in names)

    if present("pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "pytest.ini"):
        commands.append(list(_PYTEST))
    if present("package.json"):
        if present("pnpm-lock.yaml"):
            commands.append(["pnpm", "test", "--", "--silent"])
        elif present("yarn.lock"):
            commands.append(["yarn", "test", "--silent"])
        else:
            commands.append(["npm", "test", "--", "--silent"])
    if present("Cargo.toml"):
        commands.append(["cargo", "test", "-q"])
    if present("go.mod"):
        commands.append(["go", "test", "./..."])
    return commands


def _heuristic_verdict(response: str, output: str) -> TestResult:
    """Conservative verdict used when the model's reply is not JSON."""
    lowered = (response or output).lower()
    passed = ("passed" in lowered) and ("fail" not in lowered) and ("error" not in lowered)
    tail = (output or response).strip()[-_SUMMARY_TAIL:]
    summary = tail or "no output"
    if passed:
        return TestResult(passed=True, summary=summary, failures=[])
    return TestResult(
        passed=False,
        summary="Judged failed: the tester reply was not JSON and nothing clearly "
        "reported success.",
        failures=["Tester response was not valid JSON; treated as a failure.", summary],
    )


def _failed_task_ids(tasks: list[TaskModel]) -> list[str]:
    """Return the ids of tasks whose status is ``failed``."""
    return [
        str(getattr(task, "id", "?"))
        for task in tasks
        if str(getattr(task, "status", "")).lower() == "failed"
    ]


def _failed_notes(failed_ids: list[str]) -> list[str]:
    """Describe failed tasks for the ``failures`` list."""
    return [f"Task {task_id} was reported as failed." for task_id in failed_ids]


def _truncate(output: str) -> str:
    """Keep the tail of long output, where test failures are reported."""
    if len(output) <= _MAX_OUTPUT_CHARS:
        return output
    return _TRUNCATED + output[-_MAX_OUTPUT_CHARS:]


def _one_line(text: str) -> str:
    """Collapse ``text`` to a single short line."""
    collapsed = " ".join(str(text).split())
    return collapsed[:160] + ("..." if len(collapsed) > 160 else "")