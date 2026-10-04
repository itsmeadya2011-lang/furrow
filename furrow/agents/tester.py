"""Project verification for Furrow: run the real checks, then let the model explain them.

:class:`TesterAgent` is the only component that knows whether the repository actually
works. It resolves up to three commands - the test suite, the linter and the type checker
(the last two are promised by ``.kilo/FURROW.md`` and ``.kilo/agent/furrow-tester.md``),
executes them inside ``settings.resolved_workspace()`` with no shell, and only then asks
the model for a verdict.

Three properties are load bearing:

* **Detection, not guessing.** A runner is used only when its executable is on PATH
  (``shutil.which``) *and* the project carries a marker file for it. The old
  implementation returned after the first command that merely *spawned*, so a runner that
  existed but was the wrong one shadowed every later candidate.
* **A timeout is a failure.** The process is killed and the timeout is recorded. Falling
  through to the next runner is how a hanging suite used to become a green result.
* **The verdict is gated on the ground truth.** The model's JSON decides the *narrative*
  - the summary and the list of individual failures - but it can never declare success
  for a command that exited non-zero or timed out; that is overridden below.

Constructor shape is a contract with the orchestrator:
``TesterAgent(client=client, settings=settings)``. The optional ``bus=`` only adds
``TEST_RESULT`` / ``ERROR`` events to the live feed.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.events import EventType
from furrow.llm import LLMClient, LLMResponseError

if TYPE_CHECKING:
    from furrow.config import Settings
    from furrow.events import EventBus

MAX_OUTPUT_CHARS = 8_000
MAX_SUMMARY_CHARS = 600
MAX_FAILURE_CHARS = 400
MAX_CONTEXT_TASKS = 20
MAX_CONTEXT_FILES = 6
EVENT_FAILURE_LIMIT = 5
REAP_TIMEOUT = 5.0

NO_RUNNER_SUMMARY = "No test runner detected; verification skipped."
UNUSABLE_REPLY_SUMMARY = "Tester reply was not usable JSON; treated as failure"
OVERRIDE_NOTE = "command exit statuses contradict the model verdict"

# `python` is absent on many modern Linux images while `python3` is present, so an
# interpreter-prefixed candidate is retried with a name that actually resolves.
_INTERPRETERS = ("python3", "python")

_PYTEST_MARKERS = ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "conftest.py", "tests")
_PYTHON_MARKERS = _PYTEST_MARKERS + ("setup.py", "requirements.txt")
_NODE_MARKERS = ("package.json",)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One possible command plus the project files that make it meaningful."""

    argv: tuple[str, ...]
    markers: tuple[str, ...] = ()


@dataclass(slots=True)
class _Execution:
    """Ground truth about a single command: what ran, how it exited, what it printed."""

    argv: tuple[str, ...]
    command: str
    exit_code: int | None
    timed_out: bool
    output: str
    launch_error: str = ""

    @property
    def status(self) -> str:
        """Human-readable exit status for prompts and events."""
        if self.launch_error:
            return f"could not be executed: {self.launch_error}"
        return "timed out" if self.timed_out else f"exit code {self.exit_code}"

    @property
    def ok(self) -> bool:
        """True only when the command actually ran and exited 0."""
        return not self.launch_error and not self.timed_out and self.exit_code == 0


_TEST_CANDIDATES: tuple[_Candidate, ...] = (
    _Candidate(("pytest", "-q"), _PYTEST_MARKERS),
    _Candidate(("python", "-m", "pytest", "-q"), _PYTEST_MARKERS),
    _Candidate(("npm", "test", "--silent"), _NODE_MARKERS),
    _Candidate(("pnpm", "test", "--silent"), _NODE_MARKERS),
    _Candidate(("yarn", "test", "--silent"), _NODE_MARKERS),
    _Candidate(("cargo", "test", "-q"), ("Cargo.toml",)),
    _Candidate(("go", "test", "./..."), ("go.mod",)),
    _Candidate(("mvn", "test", "-q"), ("pom.xml",)),
    _Candidate(("gradle", "test", "--quiet"), ("build.gradle", "build.gradle.kts")),
)
# Only formatters are deliberately absent: the tester must never rewrite the workspace.
_LINT_CANDIDATES: tuple[_Candidate, ...] = (
    _Candidate(("ruff", "check", ".")),
    _Candidate(("python", "-m", "ruff", "check", ".")),
    _Candidate(("flake8",)),
    _Candidate(("eslint", "."), _NODE_MARKERS),
    _Candidate(("cargo", "clippy"), ("Cargo.toml",)),
)
# `mypy` gets an explicit "." target: invoked bare it exits 2 with a usage error, which
# would read as a type error. The Python type checkers require a Python project marker -
# without one `mypy .` reports "No Python files found" and exits non-zero, which would
# turn every JS-only repository into a permanent failing cycle.
_TYPECHECK_CANDIDATES: tuple[_Candidate, ...] = (
    _Candidate(("mypy", "."), _PYTHON_MARKERS),
    _Candidate(("python", "-m", "mypy", "."), _PYTHON_MARKERS),
    _Candidate(("tsc", "--noEmit"), _NODE_MARKERS),
    _Candidate(("pyright",), _PYTHON_MARKERS),
)


def _head(text: str, limit: int) -> str:
    """Return at most the first `limit` characters of `text`, marking any truncation."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars truncated]"


def _tail(text: str, limit: int) -> str:
    """Return the LAST `limit` characters of `text`, marking any truncation.

    Test runners print their summary at the end, so the tail is the useful half.
    """
    if len(text) <= limit:
        return text
    return f"[... {len(text) - limit} earlier characters omitted ...]\n{text[-limit:]}"


def _string_list(value: Any) -> list[str]:
    """Coerce `value` to a list of non-empty strings, dropping every other entry."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _combine(stdout: str, stderr: str) -> str:
    """Join the two captured streams, labelling stderr so its origin stays clear."""
    if not stderr.strip():
        return stdout
    if not stdout.strip():
        return stderr
    return f"{stdout}\n--- stderr ---\n{stderr}"


def _resolve_executable(name: str) -> str | None:
    """Return an on-PATH executable name for `name`, or None when it is not installed."""
    if shutil.which(name) is not None:
        return name
    if name == "python":
        for alias in _INTERPRETERS:
            if shutil.which(alias) is not None:
                return alias
    return None


def _has_marker(workspace: Path, markers: tuple[str, ...]) -> bool:
    """True when no marker is required, or when at least one exists in the workspace."""
    if not markers:
        return True
    return any((workspace / marker).exists() for marker in markers)


def _detect(workspace: Path, candidates: tuple[_Candidate, ...]) -> tuple[str, ...] | None:
    """Return the first candidate whose executable exists and whose markers apply.

    Existence AND applicability are both required: an installed `pytest` means nothing
    in a Rust repository, and a `package.json` means nothing without `npm`.
    """
    for candidate in candidates:
        executable = _resolve_executable(candidate.argv[0])
        if executable is None:
            continue
        if not _has_marker(workspace, candidate.markers):
            continue
        return (executable, *candidate.argv[1:])
    return None


def _result_from_payload(payload: Any) -> TestResult | None:
    """Build a `TestResult` from untrusted model JSON; None when it is unusable.

    `passed` must be a real JSON boolean. A string, a number or a missing key is
    rejected outright rather than coerced, so the fail-closed path stays fail-closed.
    """
    if not isinstance(payload, dict):
        return None
    passed = payload.get("passed")
    if not isinstance(passed, bool):
        return None
    summary = payload.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)
    failures = [
        _head(failure, MAX_FAILURE_CHARS) for failure in _string_list(payload.get("failures"))
    ]
    return TestResult(
        passed=passed,
        summary=_head(summary.strip(), MAX_SUMMARY_CHARS),
        failures=failures,
    )


async def _collect(readers: tuple[asyncio.Task[bytes], asyncio.Task[bytes]]) -> tuple[bytes, bytes]:
    """Await both pipe readers, cancelling them if a killed child kept the pipes open.

    Without this bound a grandchild process holding stdout open would hang the whole
    cycle after the test runner itself was killed.
    """
    done, pending = await asyncio.wait(set(readers), timeout=REAP_TIMEOUT)
    for reader in pending:
        reader.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    stdout = next((reader.result() for reader in readers if reader in done), b"")
    stderr = next((reader.result() for reader in readers[1:] if reader in done), b"")
    return stdout, stderr


class TesterAgent:
    """Runs the project's tests, linter and type checker, then grades the outcome."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.client: LLMClient = client if client is not None else LLMClient(settings=settings)
        self.settings: Settings = settings if settings is not None else self.client.settings
        self.bus = bus

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Verify the workspace for `goal` and return a graded result.

        Expected problems - no runner installed, a timeout, a non-zero exit, an
        unparseable model reply - are all reported inside the returned `TestResult`
        instead of raised, because the orchestrator reacts to a failure by feeding
        `TestResult.failures` back into the next cycle. Only a genuinely unexpected
        fault (bad configuration, provider auth) propagates, and the orchestrator
        isolates it into an ERROR event.
        """
        commands = self._planned_commands()
        if not commands:
            # JUDGEMENT CALL: no test runner at all means nothing was verified, which is
            # not evidence that the code is broken. Reporting a pass here is only
            # defensible because the summary says so explicitly, and `commands=[]` keeps
            # it distinguishable from a real green run in the UI. The linter and type
            # checker are deliberately NOT run alone: a verdict that certifies zero tests
            # is worse than an explicit "skipped".
            self._emit(
                EventType.LOG,
                NO_RUNNER_SUMMARY,
                commands=[],
                workspace=str(self.settings.resolved_workspace()),
            )
            return self._publish(TestResult(passed=True, summary=NO_RUNNER_SUMMARY, commands=[]))

        timeout = float(max(1, self.settings.test_timeout))
        executions = [await self._execute(argv, timeout) for argv in commands]
        hard_failures = [self._failure_line(run) for run in executions if not run.ok]
        prompt = self._compose_prompt(goal, tasks, executions)
        try:
            payload = await self.client.complete_json(
                prompt,
                system=TESTER_PROMPT,
                model=self.client.settings.tester_model,
            )
            result = _result_from_payload(payload)
        except (LLMResponseError, TypeError, ValueError) as err:
            detail = _head(f"Tester reply was not parseable JSON: {err}", MAX_FAILURE_CHARS)
            result = TestResult(
                passed=False,
                summary=_head(f"Tester reply was not parseable JSON: {err}", MAX_SUMMARY_CHARS),
                failures=[detail],
            )
        if result is None:
            result = TestResult(
                passed=False,
                summary=UNUSABLE_REPLY_SUMMARY,
                failures=[UNUSABLE_REPLY_SUMMARY],
            )
        result.commands = [run.command for run in executions]
        self._apply_ground_truth(result, hard_failures)
        broken = [run for run in executions if not run.ok]
        for run, failure in zip(broken, hard_failures, strict=True):
            self._emit(EventType.ERROR, failure, command=run.command, status=run.status)
        return self._publish(result)

    def _planned_commands(self) -> list[tuple[str, ...]]:
        """Resolve the test, lint and typecheck commands to run, in that order.

        The TEST command decides whether verification happens at all: when it cannot be
        resolved the result is an explicit "skipped" and the other two are not run on
        their own. Every command is de-duplicated, so an override that repeats a detected
        command is still executed once.
        """
        test_command = self._resolve(self.settings.test_command, _TEST_CANDIDATES)
        if test_command is None:
            return []
        commands = [test_command]
        for configured, candidates in (
            (self.settings.lint_command, _LINT_CANDIDATES),
            (self.settings.typecheck_command, _TYPECHECK_CANDIDATES),
        ):
            extra = self._resolve(configured, candidates)
            if extra is not None and extra not in commands:
                commands.append(extra)
        return commands

    def _resolve(
        self,
        configured: str | None,
        candidates: tuple[_Candidate, ...],
    ) -> tuple[str, ...] | None:
        """An explicit command first (trusted operator config), then auto-detection.

        `settings.*_command` comes from config or the environment, so it is honoured
        verbatim: split with :func:`shlex.split` and executed without a shell, which
        keeps operator input from injecting a pipeline.
        """
        if configured and configured.strip():
            return tuple(shlex.split(configured)) or None
        return _detect(self.settings.resolved_workspace(), candidates)

    async def _execute(self, argv: tuple[str, ...], timeout: float) -> _Execution:
        """Run `argv` in the workspace, without a shell, and capture its exit status.

        A timeout is recorded as a failure of THIS command rather than a reason to try
        the next candidate: falling through is how a hanging suite becomes a pass. An
        executable that cannot even be launched (a misconfigured operator override) is
        likewise a recorded failure, not an exception out of `run`.
        """
        command = shlex.join(argv)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.settings.resolved_workspace()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except OSError as err:
            return _Execution(
                argv=argv,
                command=command,
                exit_code=None,
                timed_out=False,
                output="",
                launch_error=f"{type(err).__name__}: {err}",
            )
        assert process.stdout is not None and process.stderr is not None
        readers = (
            asyncio.create_task(process.stdout.read()),
            asyncio.create_task(process.stderr.read()),
        )
        waiter = asyncio.create_task(process.wait())
        # asyncio.wait (unlike wait_for) does not cancel the pending waiter, so the
        # child keeps its exit callback and can still be killed and reaped below.
        done, _ = await asyncio.wait({waiter}, timeout=timeout)
        timed_out = waiter not in done
        if timed_out:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await asyncio.wait({waiter}, timeout=REAP_TIMEOUT)
        stdout, stderr = await _collect(readers)
        return _Execution(
            argv=argv,
            command=command,
            exit_code=process.returncode,
            timed_out=timed_out,
            output=_combine(stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")),
        )

    def _failure_line(self, run: _Execution) -> str:
        """One line naming the command and why it is a failure."""
        if run.launch_error:
            return f"{run.command} could not be executed ({run.launch_error})"
        if run.timed_out:
            return f"{run.command} timed out and was killed"
        return f"{run.command} exited with code {run.exit_code}"

    def _apply_ground_truth(self, result: TestResult, hard_failures: list[str]) -> None:
        """Override a model verdict that contradicts the recorded exit statuses.

        A non-zero exit or a timeout always wins: the model cannot talk its way out of a
        broken suite. The reverse is intentionally NOT overridden - when every command
        exited 0 but the model still reports a failure (for example "no tests ran"), that
        judgement stands, because manufacturing a failure the ground truth does not show
        would stall the cycle on nothing.
        """
        if not hard_failures or not result.passed:
            return
        result.passed = False
        known = {failure.strip() for failure in result.failures}
        result.failures = [
            *[failure for failure in hard_failures if failure.strip() not in known],
            *result.failures,
        ]
        note = f"[{OVERRIDE_NOTE}: treated as failure]"
        result.summary = f"{result.summary} {note}" if result.summary else note

    def _compose_prompt(self, goal: str, tasks: list[TaskModel], runs: list[_Execution]) -> str:
        """Build the user message: goal, touched files, real statuses, truncated output."""
        statuses = "\n".join(f"- {run.command} -> {run.status}" for run in runs)
        blocks = "\n\n".join(
            f"### {run.command}\nstatus: {run.status}\n"
            f"output:\n{_tail(run.output, MAX_OUTPUT_CHARS)}"
            for run in runs
        )
        return (
            f"Goal:\n{goal}\n\n"
            f"Tasks implemented this cycle:\n{self._task_context(tasks)}\n\n"
            f"Commands executed in the project workspace:\n{statuses}\n\n"
            f"{blocks}\n\n"
            "Decide the verdict from the exit statuses above. If ANY command exited "
            "non-zero or timed out, \"passed\" must be false and that command must "
            "appear in \"failures\" with its exit status. Report a pass only when "
            "every command exited 0.\n"
        )

    def _task_context(self, tasks: list[TaskModel]) -> str:
        """Render the cycle's tasks compactly, so the model knows what was attempted."""
        lines = [
            f"- {task.id}: {task.description} (files: {', '.join(task.files[:MAX_CONTEXT_FILES])})"
            for task in tasks[:MAX_CONTEXT_TASKS]
        ]
        return "\n".join(lines) if lines else "- (no tasks were executed)"

    def _publish(self, result: TestResult) -> TestResult:
        """Emit the TEST_RESULT event and return `result` unchanged."""
        if self.bus is not None:
            self.bus.emit(
                EventType.TEST_RESULT,
                result.summary,
                passed=result.passed,
                summary=result.summary,
                commands=list(result.commands),
                failures=result.failures[:EVENT_FAILURE_LIMIT],
                failure_count=len(result.failures),
            )
        return result

    def _emit(self, event: EventType, message: str, **data: Any) -> None:
        """Emit on the bus when one was supplied; a missing bus is not an error."""
        if self.bus is not None:
            self.bus.emit(event, message, **data)
