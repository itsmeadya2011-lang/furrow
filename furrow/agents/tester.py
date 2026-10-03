"""Test runner discovery, bounded subprocess execution and LLM verdict parsing."""

from __future__ import annotations

import asyncio
import shlex
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings

#: Wall clock ceiling for a single runner invocation, in seconds. Exceeding it is reported
#: as a timeout: a slow but legitimate suite must never be silently replaced by whatever
#: some other runner has to say about a different project.
TEST_RUN_TIMEOUT_SECONDS = 120

#: Grace period for reaping a killed runner before giving up on it.
REAP_TIMEOUT_SECONDS = 10

#: Upper bound on the characters of runner output handed to the model. A large suite
#: produces far more output than fits into a model's context window.
MAX_OUTPUT_CHARS = 8000

#: Reserved per stream for its ``[label]`` header and its truncation notice, so that the
#: labelled sections still fit inside :data:`MAX_OUTPUT_CHARS` together. Without this a
#: noisy suite loses the headers to the final trim, and the model can no longer tell which
#: stream a failure came from.
STREAM_OVERHEAD_CHARS = 64

#: Attempts made to obtain a parseable verdict: the first one plus at most one retry.
MAX_VERDICT_ATTEMPTS = 2

#: Tasks described in the prompt before the listing is cut off.
MAX_TASKS_IN_PROMPT = 10

#: Bounds on the ``failures`` list handed back to the orchestrator.
MAX_FAILURES = 20
MAX_DERIVED_FAILURES = 5
MAX_FAILURE_CHARS = 300

#: Marker files identifying a Python project whose suite is pytest.
PYTEST_MARKERS: tuple[str, ...] = ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg")
#: Marker file identifying a Node.js project.
NODE_MARKER = "package.json"
#: Lockfiles pinning which Node package manager drives ``test``.
NODE_LOCKFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pnpm-lock.yaml", ("pnpm", "test", "--", "--silent")),
    ("yarn.lock", ("yarn", "test", "--silent")),
    ("package-lock.json", ("npm", "test", "--", "--silent")),
    ("bun.lockb", ("bun", "test")),
)
#: Used for a Node project that ships no lockfile.
NODE_DEFAULT: tuple[str, ...] = ("npm", "test", "--", "--silent")
CARGO_MARKER = "Cargo.toml"
GO_MARKER = "go.mod"
#: Every marker this module knows how to recognise, used for the "nothing matched" message.
KNOWN_MARKERS: tuple[str, ...] = (*PYTEST_MARKERS, NODE_MARKER, CARGO_MARKER, GO_MARKER)

#: Prefixes written by :func:`_format_output`, stripped again before mining failures.
OUTPUT_LABELS: tuple[str, ...] = ("[stdout]", "[stderr]")
#: Case folded markers of a line worth reporting as a failure.
FAILURE_MARKERS: tuple[str, ...] = (
    "failed",
    "failure",
    "error",
    "assert",
    "traceback",
    "exception",
    "panic:",
    "not ok",
    "\u2717",
    "\u2715",
    "\u00d7",
)
#: Lines that mention a failure keyword but report the absence of failures.
BENIGN_MARKERS: tuple[str, ...] = ("no failures", "0 failed", "0 error", "no tests ran")
#: Appended to the retry prompt when the first reply was not parseable.
JSON_ONLY_INSTRUCTION = (
    "Your previous reply was not a JSON object. Reply with a single JSON object and "
    "nothing else: no markdown fences, no prose, no explanation."
)


class RunnerMissingError(RuntimeError):
    """Raised when the detected test runner is not installed."""


@dataclass(frozen=True)
class RunnerOutcome:
    """What a single runner invocation produced."""

    argv: tuple[str, ...]
    returncode: int | None
    output: str
    timed_out: bool = False


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or default_settings)
        # ``LLMClient`` resolves the fallback itself, so this is always a real ``Settings``:
        # reading ``tester_model`` or ``workspace`` off it can never hit a ``None``.
        self.settings = self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        workspace = self._workspace()
        problem = _workspace_problem(workspace)
        if problem is not None:
            return TestResult(passed=False, summary=problem, failures=[problem])

        try:
            outcome = await self._run_tests()
        except Exception as exc:  # noqa: BLE001 - reported as a failed result, never hidden
            detail = f"Test run failed: {exc}"
            return TestResult(passed=False, summary=detail, failures=[detail])

        if outcome is None:
            detail = (
                f"No recognised test runner in workspace {workspace}: "
                f"expected one of {', '.join(KNOWN_MARKERS)}."
            )
            return TestResult(passed=False, summary=detail, failures=[detail])

        return await self._verdict(goal, tasks, outcome)

    def _workspace(self) -> Path:
        """Absolute workspace directory every runner command is executed in."""
        return Path(self.settings.workspace).expanduser().resolve()

    async def _run_tests(self) -> RunnerOutcome | None:
        """Run the project's test suite once in the workspace.

        Returns ``None`` when the project type is not recognised, and raises when the
        detected runner cannot be started at all.
        """
        workspace = self._workspace()
        candidates = runner_candidates(workspace)
        if not candidates:
            return None

        missing = f"No test runner could be started for workspace {workspace}."
        for argv in candidates:
            try:
                return await _spawn_runner(argv, workspace)
            except FileNotFoundError:
                # Only a missing binary means "try the next candidate"; anything else is a
                # real problem and is reported instead of being skipped over.
                missing = f"Test runner is not installed: {shlex.join(argv)}"
                continue
            except OSError as exc:
                raise RuntimeError(f"could not run {shlex.join(argv)}: {exc}") from exc
        raise RunnerMissingError(missing)

    def _prompt(self, goal: str, tasks: Sequence[TaskModel], outcome: RunnerOutcome) -> str:
        sections = [TESTER_PROMPT, f"Goal: {goal}"]
        if tasks:
            listing = "\n".join(f"- {task.id}: {task.description}" for task in tasks[:MAX_TASKS_IN_PROMPT])
            sections.append(f"Tasks under test:\n{listing}")
        sections.append(f"Test output:\n{outcome.output}\n")
        return "\n\n".join(sections)

    async def _verdict(
        self, goal: str, tasks: Sequence[TaskModel], outcome: RunnerOutcome
    ) -> TestResult:
        prompt = self._prompt(goal, tasks, outcome)
        last_error = "the model returned no verdict"
        for attempt in range(MAX_VERDICT_ATTEMPTS):
            retry = "" if attempt == 0 else f"\n\n{JSON_ONLY_INSTRUCTION}"
            response = await self.client.complete(
                prompt + retry, model=self.settings.tester_model
            )
            try:
                return _verdict_from_payload(extract_json(response), outcome)
            except (TypeError, ValueError) as exc:
                # ``extract_json`` rejects unparseable text with a ``ValueError``; a payload
                # that parses but is not an object is rejected with a ``TypeError``. Both
                # are recoverable: the retry gets another go at it.
                last_error = str(exc)
        return _finalise(
            outcome,
            model_passed=None,
            summary=f"Model verdict was not valid JSON: {last_error}",
            failures=[],
        )


def _workspace_problem(workspace: Path) -> str | None:
    """Return why ``workspace`` cannot host a test run, or ``None`` when it can."""
    if not workspace.exists():
        return f"Workspace does not exist: {workspace}"
    if not workspace.is_dir():
        return f"Workspace is not a directory: {workspace}"
    return None


def runner_candidates(workspace: Path) -> list[tuple[str, ...]]:
    """Return the runner commands this project actually uses, most specific first.

    The runner is chosen from marker files in ``workspace`` *before* anything is executed.
    "Start whichever runner binary answers first" makes the harness report on an unrelated
    suite, and gives up on a slow suite in favour of one; an empty list means the project
    type is not recognised, and running nothing beats guessing.
    """
    if any((workspace / marker).is_file() for marker in PYTEST_MARKERS):
        # ``-m pytest`` reuses the interpreter running Furrow, so the runner exists and
        # matches the installed pytest.
        return [(sys.executable, "-m", "pytest", "-q")]
    if (workspace / NODE_MARKER).is_file():
        candidates = [
            argv for lockfile, argv in NODE_LOCKFILES if (workspace / lockfile).exists()
        ]
        if not candidates:
            candidates = [NODE_DEFAULT]
        if candidates[0] != NODE_DEFAULT:
            # Same project, same suite: only the package manager differs.
            candidates.append(NODE_DEFAULT)
        return candidates
    if (workspace / CARGO_MARKER).is_file():
        return [("cargo", "test", "-q")]
    if (workspace / GO_MARKER).is_file():
        return [("go", "test", "./...")]
    return []


async def _spawn_runner(argv: Sequence[str], workspace: Path) -> RunnerOutcome:
    """Run one runner command in ``workspace``, bounded by :data:`TEST_RUN_TIMEOUT_SECONDS`."""
    command = tuple(argv)
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(workspace),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # ``shield`` keeps the reader alive across the timeout, so a killed runner can still be
    # drained and reaped instead of leaving a zombie behind for the next cycle to race.
    reading = asyncio.ensure_future(proc.communicate())
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.shield(reading), TEST_RUN_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()
        note = (
            f"[furrow] {shlex.join(command)} timed out after {TEST_RUN_TIMEOUT_SECONDS}s and was "
            "killed; the output below is incomplete.\n\n"
        )
        try:
            stdout, stderr = await asyncio.wait_for(reading, REAP_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            # The kill could not unblock the runner; give up on its output but keep the
            # failure visible.
            stdout, stderr = b"", b""
        return RunnerOutcome(
            argv=command,
            returncode=proc.returncode,
            output=note + _format_output(stdout, stderr),
            timed_out=True,
        )
    return RunnerOutcome(
        argv=command,
        returncode=proc.returncode,
        output=_format_output(stdout, stderr),
    )


def _format_output(stdout: bytes, stderr: bytes) -> str:
    """Render runner output as labelled, length bounded text that keeps the tail.

    The two streams are labelled rather than concatenated because their interleaving is
    what makes a failure diagnosable, and each is trimmed from the front because pytest and
    most runners put the summary and the failures at the very end.
    """
    populated = [(label, raw) for label, raw in (("stdout", stdout), ("stderr", stderr)) if raw.strip()]
    share = max(1, (MAX_OUTPUT_CHARS - STREAM_OVERHEAD_CHARS * len(populated)) // max(1, len(populated)))
    sections = [f"[{label}]\n{_tail(raw.decode('utf-8', 'replace'), share).rstrip()}" for label, raw in populated]
    return _tail("\n".join(sections), MAX_OUTPUT_CHARS)


def _tail(text: str, limit: int) -> str:
    """Return the last ``limit`` characters of ``text``, flagging what was dropped."""
    if len(text) <= limit:
        return text
    return f"... [{len(text) - limit} earlier characters omitted] ...\n{text[-limit:]}"


def _verdict_from_payload(data: Any, outcome: RunnerOutcome) -> TestResult:
    """Validate a decoded model verdict into a :class:`TestResult`."""
    if not isinstance(data, dict):
        raise TypeError(f"verdict must be a JSON object, got {type(data).__name__}")
    return _finalise(
        outcome,
        model_passed=_coerce_bool(data.get("passed")),
        summary=str(data.get("summary") or ""),
        failures=_coerce_failures(data.get("failures")),
    )


def _finalise(
    outcome: RunnerOutcome,
    *,
    model_passed: bool | None,
    summary: str,
    failures: Iterable[Any],
) -> TestResult:
    """Combine the model's claim with the authoritative runner exit code."""
    passed = _resolve_passed(model_passed, outcome)
    notes = [note for note in (_run_note(outcome),) if note]
    text = summary.strip()
    text = f"{text} ({notes[0]})" if text and notes else text or " ".join(notes)
    if passed:
        return TestResult(passed=True, summary=text, failures=[])

    actionable = [_shorten(str(item).strip()) for item in failures if str(item).strip()]
    if not actionable:
        # The orchestrator rebuilds its focus from ``failures``; an empty list would leave
        # it with nothing to act on.
        actionable = _failures_from_output(outcome.output)
    if not actionable:
        actionable = [text or "tests failed"]
    return TestResult(passed=False, summary=text, failures=actionable[:MAX_FAILURES])


def _resolve_passed(model_passed: bool | None, outcome: RunnerOutcome) -> bool:
    """Decide the verdict, with the runner's exit code as the authoritative signal."""
    if outcome.timed_out or outcome.returncode != 0:
        return False
    # An unusable verdict falls back to the exit code alone; a model that does report a
    # failure on a green exit code is believed, because the exit code of a suite that
    # merely collected nothing says nothing about the goal.
    return True if model_passed is None else model_passed


def _run_note(outcome: RunnerOutcome) -> str:
    """Describe how the runner itself ended, for the summary."""
    if outcome.timed_out:
        return f"{shlex.join(outcome.argv)} timed out after {TEST_RUN_TIMEOUT_SECONDS}s"
    if outcome.returncode:
        return f"{shlex.join(outcome.argv)} exited with code {outcome.returncode}"
    return ""


def _coerce_bool(value: Any) -> bool | None:
    """Read a model supplied verdict flag, tolerating the shapes models actually emit."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "pass", "passed"}:
            return True
        if lowered in {"false", "no", "fail", "failed"}:
            return False
    return None


def _coerce_failures(value: Any) -> list[str]:
    """Read a model supplied failure list, tolerating a bare string."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [_shorten(str(item)) for item in items if str(item).strip()]


def _failures_from_output(output: str) -> list[str]:
    """Mine actionable failure lines out of runner output, most recent last."""
    seen: list[str] = []
    for raw in output.splitlines():
        line = _strip_labels(raw).strip()
        lowered = line.lower()
        if not line or any(benign in lowered for benign in BENIGN_MARKERS):
            continue
        if not any(marker in lowered for marker in FAILURE_MARKERS) or line in seen:
            continue
        seen.append(_shorten(line))
    return seen[-MAX_DERIVED_FAILURES:]


def _strip_labels(line: str) -> str:
    """Remove a stream label written by :func:`_format_output`."""
    for label in OUTPUT_LABELS:
        if line.startswith(label):
            return line[len(label) :]
    return line


def _shorten(text: str, limit: int = MAX_FAILURE_CHARS) -> str:
    """Collapse whitespace and clip a single failure string to ``limit`` characters."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."