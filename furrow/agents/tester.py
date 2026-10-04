"""Tester agent: runs the project's test suite, then asks an LLM for a verdict."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.planner import extract_json_object, truncate_for_log
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient, LLMResponseError

if TYPE_CHECKING:
    from furrow.config import Settings

#: Overall wall-clock budget for a single ``_run_tests()`` sweep across every
#: candidate runner.  Each candidate used to get its own 120 s timeout, so a fully
#: unresponsive machine stalled one cycle for 7 * 120 s = 840 s.  The sweep is now
#: bounded end to end: a candidate only ever gets the smaller of its own slice and
#: whatever is left of this budget.
TEST_SUITE_TIMEOUT_SECONDS = 90.0

#: Maximum time one candidate runner may consume, itself capped by the sweep budget.
TEST_RUNNER_TIMEOUT_SECONDS = 60.0

#: Grace period for reaping a killed child before giving up on ``proc.wait()``.
TEST_REAP_TIMEOUT_SECONDS = 5.0

#: Sentinels.  "Timed out" and "no runner" used to share one string, which is why a
#: stalled environment was reported as "No test runner found.".
NO_TEST_RUNNER_MESSAGE = "No test runner found."
TEST_TIMED_OUT_MESSAGE = "Test run timed out."

#: Test output embedded in the classifier prompt.  The tail is kept because that is
#: where runner summaries (the part worth reading) live.
MAX_OUTPUT_CHARS = 8_000

#: Entries written to ``TestResult.failures``; the orchestrator turns them into the
#: next cycle's goal, so they must carry information instead of being empty.
MAX_FAILURE_CHARS = 1_000

#: "0 failed" must not be read as a failure, but "1 failed" must be.
_ZERO_COUNT_RE = re.compile(r"\b0\s+(failed|failing|failures|errors?|passed|tests?|warnings?)\b")

#: Checked before the pass markers, so "No tests passed" is a failure.
_FAILURE_MARKERS: tuple[str, ...] = (
    "failed",
    "failing",
    "failure",
    "failures",
    "error:",
    "errors:",
    "traceback",
    "exception",
    "assertionerror",
    "no tests ran",
    "no tests passed",
    "no tests collected",
    "not passed",
    "did not pass",
    "usage error",
    "internal error",
    "segmentation fault",
    "not ok",
)

_PASS_MARKERS: tuple[str, ...] = ("passed", "passing", "success", "succeeded")


@dataclass(frozen=True)
class TestRunner:
    """A candidate test command and the exit codes that mean "tests actually ran".

    Attributes:
        name: Label used in messages.
        argv: Executable plus arguments.
        ran_exit_codes: Exit codes that represent a real test outcome.  Anything
            else (usage error, missing toolchain, runner internal error, or a
            manager that refused to start) means the candidate falls through to the
            next one instead of being reported as the project's test result.
        manifest: Project file that must exist for the runner to be applicable, so
            ``npm test`` is not launched in a Python-only tree.
    """

    name: str
    argv: tuple[str, ...]
    ran_exit_codes: frozenset[int]
    manifest: str | None = None


# Runner selection order.  Exit-code rules:
#   pytest  0 passed, 1 tests failed (both real outcomes); 2 interrupted,
#           3 internal error, 4 usage error, 5 nothing collected -> fall through.
#   npm / pnpm / yarn  0 ok, 1 script/test failure; anything else is the CLI
#           itself failing (e.g. 127 missing script) -> fall through.
#   cargo   0 ok, 101 tests failed; 1 is a cargo/compile error -> fall through.
#   go      0 ok, 1 at least one test failed (guarded by the go.mod check, since
#           "no go.mod" is also exit 1).
TEST_RUNNERS: tuple[TestRunner, ...] = (
    TestRunner("pytest", ("pytest", "-q"), frozenset({0, 1})),
    TestRunner("python -m pytest", (sys.executable, "-m", "pytest", "-q"), frozenset({0, 1})),
    TestRunner("npm test", ("npm", "test", "--", "--silent"), frozenset({0, 1}), "package.json"),
    TestRunner("pnpm test", ("pnpm", "test", "--", "--silent"), frozenset({0, 1}), "package.json"),
    TestRunner("yarn test", ("yarn", "test", "--silent"), frozenset({0, 1}), "package.json"),
    TestRunner("cargo test", ("cargo", "test", "-q"), frozenset({0, 101}), "Cargo.toml"),
    TestRunner("go test", ("go", "test", "./..."), frozenset({0, 1}), "go.mod"),
)


@dataclass(frozen=True)
class TestRunOutcome:
    """What actually happened when the tester tried to run the test suite."""

    output: str
    runner: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    notes: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        if self.runner is not None:
            return f"{self.runner} exited {self.exit_code}"
        if self.timed_out:
            return TEST_TIMED_OUT_MESSAGE
        return NO_TEST_RUNNER_MESSAGE


@dataclass(frozen=True)
class _RunnerResult:
    """Result of a single runner invocation (``timed_out`` means unusable)."""

    output: str
    exit_code: int | None = None
    timed_out: bool = False


def judge_passed(output: str) -> bool:
    """Conservatively infer a pass/fail verdict from raw runner output.

    ``passed = "passed" in output`` is wrong in both directions: "No tests passed"
    reads as success and a sentence like "tests did not pass" reads as failure.

    Returns:
        True only when the text carries a success marker and no failure marker.
        Empty, truncated or otherwise ambiguous output is reported as a failure so
        the caller keeps the raw output instead of silently claiming success.
    """
    stripped = output.strip()
    if not stripped:
        return False
    text = _ZERO_COUNT_RE.sub(" ", stripped.lower())
    if any(marker in text for marker in _FAILURE_MARKERS):
        return False
    return any(marker in text for marker in _PASS_MARKERS)


def describe_tasks(
    tasks: list[TaskModel], *, limit: int = 10, description_chars: int = 200
) -> str:
    """Render the planned tasks (ids plus truncated descriptions) for a prompt."""
    if not tasks:
        return "(no tasks were planned for this cycle)"
    lines: list[str] = []
    for task in tasks[:limit]:
        description = " ".join(task.description.split())
        if len(description) > description_chars:
            description = description[:description_chars].rstrip() + "..."
        lines.append(f"- [{task.id}] {description}")
    if len(tasks) > limit:
        lines.append(f"- ... and {len(tasks) - limit} more task(s)")
    return "\n".join(lines)


def clip_output(output: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the tail of ``output`` so the runner summary survives truncation."""
    if len(output) <= limit:
        return output
    marker = f"... [truncated, {len(output)} chars total, showing the last {limit}]\n"
    return marker + output[-limit:]


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the runner's whole process group, then reap the child.

    ``proc.kill()`` alone signals only the direct child, so ``npm test`` / ``go
    test`` grandchildren survive; and killing without awaiting leaves a zombie and
    an unreaped pipe transport behind.
    """
    try:
        # start_new_session=True made the child a process group leader, so its pid
        # is the pgid and the kill reaches every descendant.
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        # Group already gone, or not ours to signal: fall back to the direct child.
        try:
            proc.kill()
        except OSError:
            pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=TEST_REAP_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        pass  # Child is wedged in uninterruptible I/O; nothing further to do here.


class TesterAgent:
    """Runs the project's tests and turns the result into a :class:`TestResult`.

    Args:
        client: Injected LLM client. When omitted a default one is built.
        settings: Configuration used to build the default client; ``settings.workspace``
            is the directory tests run in. Ignored - with a warning - when it
            conflicts with ``client.settings``; the explicitly passed ``client``
            always wins, as it did before.
    """

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is None:
            self.client = LLMClient() if settings is None else LLMClient(settings=settings)
        else:
            client_settings = getattr(client, "settings", None)
            if (
                settings is not None
                and client_settings is not None
                and settings is not client_settings
            ):
                warnings.warn(
                    "TesterAgent got both client= and settings=; the client's own settings "
                    "take precedence and the settings argument is ignored.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.client = client
        resolved = getattr(self.client, "settings", None)
        if resolved is None:
            resolved = settings if settings is not None else default_settings
        self.settings: Settings = resolved

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the test suite and classify the outcome against ``goal``/``tasks``."""
        try:
            outcome = await self._run_tests()
        except Exception as e:
            # run() must not raise: the orchestrator builds its next goal from this
            # result, and an exception here silently ends the cycle.
            message = truncate_for_log(f"Test execution error: {e!r}")
            return TestResult(passed=False, summary=message, failures=[message])

        prompt = self._build_prompt(goal, tasks, outcome)
        response = await self.client.complete(prompt, model=self.settings.tester_model)
        try:
            data = extract_json_object(response, context="Tester response")
            return TestResult(**data)
        except (LLMResponseError, ValueError, TypeError):
            return self._fallback_verdict(response, outcome)

    def _build_prompt(self, goal: str, tasks: list[TaskModel], outcome: TestRunOutcome) -> str:
        sections = [
            TESTER_PROMPT,
            "",
            f"Goal: {goal}",
            "",
            "Tasks that were supposed to be implemented in this cycle:",
            describe_tasks(tasks),
            "",
            f"Test run: {outcome.summary}",
        ]
        sections.extend(f"Note: {note}" for note in outcome.notes)
        sections.extend(["", "Test output:", clip_output(outcome.output)])
        return "\n".join(sections)

    def _fallback_verdict(self, response: str, outcome: TestRunOutcome) -> TestResult:
        """Heuristic verdict for a classifier reply that is not a JSON object."""
        # A runner that never produced a result, or raw output that shows failures,
        # outranks whatever the classifier's prose claims.
        passed = (
            outcome.runner is not None
            and judge_passed(outcome.output)
            and judge_passed(response)
        )
        summary = (
            f"Classifier reply was not valid JSON ({outcome.summary}); "
            f"heuristic verdict: {'passed' if passed else 'failed'}."
        )
        if passed:
            return TestResult(passed=True, summary=summary)
        # Never hand back an empty failure list: the orchestrator turns it into the
        # literal goal "Fix failing tests:\n" and the cycle loses all context.
        failures = [
            truncate_for_log(
                f"Test output ({outcome.summary}): {outcome.output}", MAX_FAILURE_CHARS
            ),
            truncate_for_log(f"Classifier reply: {response}", MAX_FAILURE_CHARS),
        ]
        return TestResult(passed=False, summary=summary, failures=failures)

    async def _run_tests(self) -> TestRunOutcome:
        """Run the first candidate runner that actually executed the test suite."""
        cwd, notes = self._resolve_cwd()
        deadline = time.monotonic() + TEST_SUITE_TIMEOUT_SECONDS
        hung: list[str] = []

        for runner in TEST_RUNNERS:
            blocking_manifest = self._missing_manifest(runner, cwd)
            if blocking_manifest is not None:
                notes.append(f"{runner.name} skipped: {blocking_manifest} not found.")
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                notes.append(
                    f"{TEST_SUITE_TIMEOUT_SECONDS:.0f}s test budget exhausted before "
                    f"{runner.name} could start."
                )
                return TestRunOutcome(
                    output=TEST_TIMED_OUT_MESSAGE, timed_out=True, notes=tuple(notes)
                )
            result = await self._execute(runner, min(TEST_RUNNER_TIMEOUT_SECONDS, remaining), cwd)
            if result is None:
                notes.append(f"{runner.name} unavailable (command not found or not executable).")
                continue
            if result.timed_out:
                hung.append(runner.name)
                notes.append(f"{runner.name} timed out; killing its process group.")
                continue
            if result.exit_code in runner.ran_exit_codes:
                return TestRunOutcome(
                    output=result.output,
                    runner=runner.name,
                    exit_code=result.exit_code,
                    timed_out=bool(hung),
                    notes=tuple(notes),
                )
            notes.append(
                f"{runner.name} exited {result.exit_code} without running tests "
                f"(not in {sorted(runner.ran_exit_codes)}); trying the next runner."
            )

        if hung:
            notes.append(f"runners that timed out: {', '.join(hung)}.")
            return TestRunOutcome(
                output=TEST_TIMED_OUT_MESSAGE, timed_out=True, notes=tuple(notes)
            )
        return TestRunOutcome(output=NO_TEST_RUNNER_MESSAGE, notes=tuple(notes))

    def _resolve_cwd(self) -> tuple[str | None, list[str]]:
        """Return the directory to run tests in, plus notes about any fallback."""
        workspace = getattr(self.settings, "workspace", None)
        if workspace is None:
            return None, []
        path = Path(workspace)
        if path.is_dir():
            return str(path), []
        # No cwd= means the server's own working directory; say so out loud.
        return None, [
            f"Configured workspace {path} does not exist; "
            f"tests ran in the process working directory instead."
        ]

    @staticmethod
    def _missing_manifest(runner: TestRunner, cwd: str | None) -> str | None:
        """Return the manifest path that disqualifies ``runner``, if any."""
        if runner.manifest is None:
            return None
        base = Path(cwd) if cwd is not None else Path.cwd()
        candidate = base / runner.manifest
        return None if candidate.exists() else str(candidate)

    async def _execute(
        self, runner: TestRunner, timeout: float, cwd: str | None
    ) -> _RunnerResult | None:
        """Run one candidate, or return ``None`` when its binary is unusable."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *runner.argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                # Own process group: a timeout must reach `npm test` grandchildren too.
                start_new_session=True,
            )
        except OSError:
            # FileNotFoundError/NotADirectoryError/PermissionError - the only
            # expected spawn failures.  Catching Exception here also swallowed
            # asyncio errors and made them look like "no runner found".
            return None
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await _kill_process_group(proc)
            return _RunnerResult(
                output=f"{runner.name} timed out after {timeout:.0f}s", timed_out=True
            )
        # errors="replace": odd/binary output must not raise UnicodeDecodeError and
        # be reported as "no test runner found".
        output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
        return _RunnerResult(output=output, exit_code=proc.returncode)