from __future__ import annotations

import asyncio
import re
import shlex
import shutil
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from furrow.agents.jsonutil import parse_json_model
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from furrow.config import Settings


#: Captured output is trimmed to this many characters before it enters a prompt.
#: A suite with thousands of passing tests otherwise costs a huge prompt on every
#: single cycle, and the informative part of the output is at the end.
MAX_OUTPUT_CHARS = 8000

#: pytest's documented exit code for "collected nothing", as opposed to failures.
_PYTEST_NO_TESTS_EXIT = 5

#: Output patterns meaning "the runner started and executed no tests".  These mean
#: "this is the wrong runner", not "the suite is broken".
NO_TESTS_PATTERNS = (
    re.compile(r"no tests?\s+(?:were\s+|was\s+)?(?:collected|ran|found|executed)", re.I),
    re.compile(r"no test files?\s+found", re.I),
    re.compile(r"no tests?\s+to run", re.I),
    re.compile(r"\[no tests\]", re.I),
    re.compile(r"^\s*running 0 tests\b", re.I | re.M),
    re.compile(r"0 tests?\s+collected", re.I),
    re.compile(r"error: no tests? found", re.I),
)

#: Phrasings that deny a successful outcome even though they contain a success
#: word: "not passed", "did not pass", "tests did not succeed", "0 passed".
#: These are checked *before* the success markers, because a bare keyword search
#: scores "not passed" as a pass — a false pass halts the loop on a red suite.
_NEGATION_PATTERNS = (
    # "... not passed", "... did not pass", "... not passing"
    re.compile(r"\bnot\s+(?:\w+\s+){0,2}?pass(?:es|ed|ing)?\b"),
    re.compile(r"\bnever\s+(?:\w+\s+){0,2}?pass(?:es|ed|ing)?\b"),
    # "didn't pass", "doesn't pass", "isn't passing"
    re.compile(r"\bn[o']t\s+(?:\w+\s+){0,2}?pass(?:es|ed|ing)?\b"),
    # "... not succeeded", "did not succeed", "didn't succeed"
    re.compile(r"\b(?:not|never)\s+(?:\w+\s+){0,2}?succeed(?:s|ed|ing)?\b"),
    re.compile(r"\bn[o']t\s+(?:\w+\s+){0,2}?succeed(?:s|ed|ing)?\b"),
    # "... not ok", "isn't okay", "not successful"
    re.compile(r"\b(?:not|never)\s+(?:\w+\s+){0,2}?(?:ok|okay|successful)\b"),
    re.compile(r"\bn[o']t\s+(?:\w+\s+){0,2}?(?:ok|okay|successful)\b"),
    re.compile(r"\bun-?successful\w*\b"),
    # "0 passed", "zero passed", "none passed", "nothing passed", "no tests passed"
    re.compile(r"\b(?:0|zero|none|nothing|no\s+\w+)\s+(?:\w+\s+){0,2}?pass(?:es|ed|ing)?\b"),
)

#: Checked before any success marker, so "0 passed, 12 failed" and "cannot
#: import x" can never score as a pass.
_FAILURE_MARKERS = ("fail", "error", "traceback", "assertionerror", "cannot")
_SUCCESS_MARKERS = ("passed", "succeed", "ok")

_PYTEST_MARKERS = ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini")
_MARKER_FILES = (*_PYTEST_MARKERS, "package.json", "Cargo.toml", "go.mod")

_PYTEST_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
)
_CANDIDATES: tuple[tuple[str, ...], ...] = (
    *_PYTEST_CANDIDATES,
    ("npm", "test", "--", "--silent"),
    ("pnpm", "test", "--", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)


def truncate_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Trim *text* to *limit* characters, keeping the **tail**.

    Test runners put failures at the end of their output, so a head truncation
    keeps the part that matters least.  The number of dropped characters is
    stated in the marker so the tester model is not misled into thinking it is
    looking at the whole run.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"... [{dropped} earlier characters omitted]\n{text[-limit:]}"


def _is_pytest(command: Sequence[str]) -> bool:
    return any(part.endswith("pytest") for part in command)


def _node_candidate(workspace: Path) -> list[str]:
    if (workspace / "pnpm-lock.yaml").exists():
        return ["pnpm", "test", "--", "--silent"]
    if (workspace / "yarn.lock").exists():
        return ["yarn", "test", "--silent"]
    return ["npm", "test", "--", "--silent"]


def detect_candidates(workspace: Path) -> list[list[str]]:
    """Return candidate test commands for *workspace*, most likely first.

    Detection is driven by marker files rather than by "which binary happens to
    exist": ``pytest`` is installed in nearly every Python environment, so an
    existence-first sweep over a Node, Rust or Go project returns pytest's
    "no tests collected" output and never reaches the real runner.  Everything
    that was not detected is still appended, so a wrong detection costs one
    wasted invocation instead of a missed test run.
    """
    present = {name for name in _MARKER_FILES if (workspace / name).exists()}
    detected: list[list[str]] = []
    if present & set(_PYTEST_MARKERS):
        detected.extend(list(cmd) for cmd in _PYTEST_CANDIDATES)
    if "package.json" in present:
        detected.append(_node_candidate(workspace))
    if "Cargo.toml" in present:
        detected.append(["cargo", "test", "-q"])
    if "go.mod" in present:
        detected.append(["go", "test", "./..."])

    ordered: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for cmd in [*detected, *(list(candidate) for candidate in _CANDIDATES)]:
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            ordered.append(cmd)
    return ordered


@dataclass(frozen=True)
class RunAttempt:
    """Outcome of a single test-command invocation."""

    command: list[str]
    returncode: int | None
    output: str
    timed_out: bool = False
    error: str | None = None

    @property
    def label(self) -> str:
        return shlex.join(self.command)

    @property
    def collected_nothing(self) -> bool:
        """True when the runner executed but found no tests to run.

        That identifies the wrong runner and means "try the next candidate"; it is
        explicitly not a test failure.
        """
        if self.timed_out or self.error is not None or self.returncode is None:
            return False
        if _is_pytest(self.command) and self.returncode == _PYTEST_NO_TESTS_EXIT:
            return True
        return any(pattern.search(self.output) for pattern in NO_TESTS_PATTERNS)


@dataclass(frozen=True)
class TestRunOutcome:
    """What happened when the tester tried to execute the project's suite.

    *command*, *returncode* and *output* describe the invocation that was
    accepted as the real test run; *output* is already truncated for prompt use.
    *notes* explains every candidate that was skipped or rejected, so a prompt
    built from a degenerate run is still diagnosable.
    """

    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    output: str = ""
    ran: bool = False
    timed_out: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def exit_status(self) -> str:
        if self.timed_out:
            return "no exit status: the command timed out and was killed"
        if self.returncode is None:
            return "no exit status: the command did not report one"
        return str(self.returncode)

    @property
    def label(self) -> str:
        return shlex.join(self.command) if self.command else "(no command was run)"

    def detail(self) -> str:
        """All of the run information as one prompt-sized block."""
        parts = [f"Command: {self.label}", f"Exit status: {self.exit_status}"]
        if self.notes:
            parts.append("Notes:\n" + "\n".join(f"- {note}" for note in self.notes))
        if self.output.strip():
            parts.append("Output (tail; earlier lines may be elided):\n" + self.output)
        return "\n\n".join(parts)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        # Resolve settings once so the class does not reach through
        # self.client.settings for every field on every call.
        self.settings = self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the project's test suite and grade it with the tester model.

        *tasks* is accepted for signature compatibility with the other agents and
        is not used for the judgement: the suite itself is the signal.

        Never raises for a broken test run or an unparseable model response —
        both come back as ``TestResult(passed=False, ...)`` with whatever output
        was captured, so the orchestrator feeds the failure to the planner as
        feedback instead of losing the cycle.
        """
        outcome = await self._run_tests()
        if not outcome.ran:
            detail = "\n".join(outcome.notes) or "No test runner found."
            return TestResult(passed=False, summary=detail, failures=[detail])

        prompt = "\n\n".join([TESTER_PROMPT, f"Goal: {goal}", "Test run:", outcome.detail()])
        response = await self.client.complete(prompt, model=self.settings.tester_model)
        try:
            return parse_json_model(response, TestResult)
        except ValueError:
            return _heuristic_result(response)

    async def _run_tests(self) -> TestRunOutcome:
        """Execute the suite once and report what happened.

        ``settings.test_command`` wins when set (split with :func:`shlex.split`,
        never a shell), otherwise the runner is detected from the project's marker
        files.  ``settings.test_timeout`` is the budget for the *whole* sweep: each
        candidate gets the remaining time and no new candidate starts once the
        budget is gone.  Everything runs with ``cwd=settings.workspace``; that is
        the only filesystem path the tester honours, so a suite cannot be pointed
        outside the configured workspace.
        """
        started = time.monotonic()
        workspace = self.settings.workspace
        override = (self.settings.test_command or "").strip()
        candidates = [shlex.split(override)] if override else detect_candidates(workspace)

        notes: list[str] = []
        rejections = 0
        last: RunAttempt | None = None
        try:
            for cmd in candidates:
                remaining = self.settings.test_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    notes.append(
                        f"Stopped after {rejections} rejected invocation(s): the "
                        f"{self.settings.test_timeout:g}s test budget is exhausted."
                    )
                    break
                if shutil.which(cmd[0]) is None:
                    notes.append(f"{cmd[0]!r} is not on PATH; skipped without spawning.")
                    continue
                attempt = await self._run_command(cmd, workspace, remaining)
                last = attempt
                rejections += 1
                if attempt.error is not None:
                    notes.append(f"{attempt.label} could not start: {attempt.error}")
                    continue
                if attempt.timed_out:
                    notes.append(f"{attempt.label} timed out after {remaining:g}s.")
                    continue
                if attempt.collected_nothing:
                    notes.append(
                        f"{attempt.label} collected no tests "
                        f"(exit status {attempt.returncode}); trying the next runner."
                    )
                    continue
                return TestRunOutcome(
                    command=attempt.command,
                    returncode=attempt.returncode,
                    output=truncate_output(attempt.output),
                    ran=True,
                    notes=notes,
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # the sweep itself blew up; keep what we captured
            detail = (
                f"Test run aborted after {type(e).__name__}: {e}. "
                f"Last command: {last.label if last else '(none)'}. "
                f"Output captured before the failure:\n"
                f"{truncate_output(last.output) if last else '(nothing captured)'}"
            )
            notes.append(detail)
            return TestRunOutcome(
                command=last.command if last else [],
                returncode=last.returncode if last else None,
                output=detail,
                ran=last is not None,
                notes=notes,
            )

        notes.append("No test runner produced a result.")
        if last is not None and last.output.strip():
            notes.append(truncate_output(last.output))
        return TestRunOutcome(
            command=last.command if last else [],
            returncode=last.returncode if last else None,
            ran=False,
            notes=notes,
        )

    async def _run_command(self, command: Sequence[str], cwd: Path, timeout: float) -> RunAttempt:
        """Spawn one candidate, bounded by *timeout* seconds."""
        cmd = list(command)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            # Includes FileNotFoundError (vanished between which() and spawn) and
            # PermissionError (unexecutable file).  Record why instead of
            # pretending no runner exists.
            return RunAttempt(
                command=cmd, returncode=None, output="", error=f"{type(e).__name__}: {e}"
            )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            # kill() only signals: without wait() the child is never reaped and its
            # pipes stay open across the next cycle.
            await proc.wait()
            return RunAttempt(command=cmd, returncode=None, output="", timed_out=True)
        return RunAttempt(
            command=cmd,
            returncode=proc.returncode,
            # errors="replace": a stray non-UTF-8 byte must not raise out of the tester.
            output=stdout.decode(errors="replace") + stderr.decode(errors="replace"),
        )


def _heuristic_result(response: str) -> TestResult:
    """Grade a non-JSON tester response conservatively.

    Order of judgement, first match wins:

    1. a denial of success (:data:`_NEGATION_PATTERNS`) — "not passed", "did not
       pass", "tests did not succeed", "0 passed".  These carry a success word,
       so they must be ruled out before the success markers are consulted;
    2. an explicit failure word (:data:`_FAILURE_MARKERS`);
    3. a success word (:data:`_SUCCESS_MARKERS`) with nothing against it.

    A response that matches none of these is scored as a failure.

    Limits: this is keyword matching, so it has no notion of scope.  A response
    that mentions a failure while reporting overall success ("all passed, though
    the skipped test could not import X") is scored as a failure.  Both errors
    push towards "keep working" rather than "declare victory", which is the
    safer direction for an unattended loop: a false pass halts it, a false fail
    costs one cycle.
    """
    text = response.lower()
    negated = any(pattern.search(text) for pattern in _NEGATION_PATTERNS)
    failed = [marker for marker in _FAILURE_MARKERS if marker in text]
    succeeded = [marker for marker in _SUCCESS_MARKERS if marker in text]
    passed = not negated and not failed and bool(succeeded)
    summary = _first_line(response) or "Tester returned an empty or unparseable response"
    failures = [] if passed else [truncate_output(response.strip(), 2000) or summary]
    return TestResult(passed=passed, summary=summary, failures=failures)


def _first_line(text: str) -> str:
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""
