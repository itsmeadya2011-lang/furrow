from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_TEST_TIMEOUT_SECONDS = 120.0
REAP_TIMEOUT_SECONDS = 5.0
PYTEST_NO_TESTS_COLLECTED = 5
MAX_FAILURES = 20
MAX_FAILURE_LINE_CHARS = 400
MAX_SUMMARY_TAIL_LINES = 5
MAX_FALLBACK_TAIL_LINES = 25
MAX_FALLBACK_TAIL_CHARS = 2000
MAX_PROMPT_OUTPUT_CHARS = 12000

TEST_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("pytest", "-q"),
    ("python3", "-m", "pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("npm", "test", "--", "--silent"),
    ("pnpm", "test", "--", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)

_FAILURE_LINE_PATTERNS = (
    re.compile(r"^(?:FAILED|ERROR)\b"),
    re.compile(r"^--- FAIL:"),
    re.compile(r"^not ok\b"),
    re.compile(r"^[✗✖×]"),
    re.compile(r"^E\s+\S"),
    re.compile(r"\bAssertionError\b"),
    re.compile(r"^panic: "),
    re.compile(r"panicked at "),
)

_FENCE_RE = re.compile(r"```[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL)


class TestRun(NamedTuple):
    """Structured outcome of a single test-runner invocation.

    ``found`` is False (and ``command`` is empty) when no runner binary exists. ``timed_out``
    is True when the runner was killed for exceeding its budget. The exit code is the only
    source of truth for the verdict.
    """

    command: tuple[str, ...] = ()
    returncode: int | None = None
    output: str = ""
    timed_out: bool = False
    found: bool = False
    timeout_seconds: float | None = None

    @property
    def command_line(self) -> str:
        return " ".join(self.command)

    @property
    def passed(self) -> bool:
        return self.found and not self.timed_out and self.returncode == 0


def _decode(raw: bytes | None) -> str:
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"...[{len(text) - limit} earlier characters omitted]...\n{text[-limit:]}"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _output_tail(output: str, max_lines: int) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _extract_failures(output: str) -> list[str]:
    """Collect failure-looking lines from real test output."""
    failures: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or len(line) > MAX_FAILURE_LINE_CHARS:
            continue
        if any(pattern.search(line) for pattern in _FAILURE_LINE_PATTERNS):
            failures.append(line)
        if len(failures) >= MAX_FAILURES:
            logger.warning("Failure list truncated at %d entries", MAX_FAILURES)
            break
    return _dedupe(failures)


def _load_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction for optional LLM enrichment. Never raises."""
    stripped = text.strip()
    candidates: list[str] = [stripped] if stripped else []
    candidates.extend(block.strip() for block in _FENCE_RE.findall(stripped))
    for start, char in enumerate(stripped):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            current = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start : index + 1])
                    break
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    """Kill the child and wait for it so no zombie or open transport is left behind."""
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=REAP_TIMEOUT_SECONDS)


async def _communicate(proc: asyncio.subprocess.Process, timeout: float) -> tuple[bytes, bytes, bool]:
    """Wait for ``proc``, returning ``(stdout, stderr, timed_out)``.

    On timeout the child is killed, reaped, and whatever output it produced before dying is
    returned instead of being discarded.
    """
    comm = asyncio.ensure_future(proc.communicate())
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.shield(comm), timeout=timeout)
        return stdout, stderr, False
    except (asyncio.TimeoutError, TimeoutError):
        await _kill_and_reap(proc)
    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError, TimeoutError):
        partial_stdout, _ = await asyncio.wait_for(comm, timeout=REAP_TIMEOUT_SECONDS)
        return partial_stdout, b"", True
    return b"", b"", True


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the project's tests and report a verdict driven by the runner's exit code."""
        task_context, task_problem = self._task_context(tasks)
        try:
            run = await self._run_tests()
        except OSError as exc:
            detail = f"Could not start the test runner: {exc}"
            logger.warning("%s", detail)
            return TestResult(passed=False, summary=detail, failures=[detail])

        if not run.found:
            detail = (
                "No test runner was found in the workspace "
                f"{self._workspace()}; tried {', '.join(cmd[0] for cmd in TEST_COMMANDS)}. "
                "The suite never ran, so the goal cannot be verified."
            )
            logger.warning("%s", detail)
            return TestResult(passed=False, summary=detail, failures=[detail])

        if run.timed_out:
            budget = run.timeout_seconds or DEFAULT_TEST_TIMEOUT_SECONDS
            detail = (
                f"`{run.command_line}` timed out after {budget:g}s in {self._workspace()} and was "
                "killed. A hung suite counts as a failure; no other runner is attempted."
            )
            logger.warning("%s", detail)
            return TestResult(
                passed=False,
                summary=detail,
                failures=_dedupe([detail, *_extract_failures(run.output)])[:MAX_FAILURES],
            )

        failures = _extract_failures(run.output)
        if not failures:
            failures = [self._tail_fallback(run)]

        if run.returncode == PYTEST_NO_TESTS_COLLECTED:
            detail = (
                f"`{run.command_line}` exited with code {PYTEST_NO_TESTS_COLLECTED} in "
                f"{self._workspace()}: no tests were collected, so nothing was verified. "
                "Add or update tests that cover this change."
            )
            logger.warning("%s", detail)
            return TestResult(
                passed=False,
                summary=detail,
                failures=_dedupe([detail, *failures])[:MAX_FAILURES],
            )

        if run.passed and task_problem is None:
            summary = self._success_summary(run)
            logger.info("Tests passed: %s", summary)
            return TestResult(passed=True, summary=summary, failures=[])

        # Green suite, but nothing was actually implemented: treat as unverified.
        if run.passed:
            logger.warning("%s", task_problem)
            summary = f"{self._success_summary(run)} - however, {task_problem}"
            return TestResult(
                passed=False,
                summary=summary,
                failures=_dedupe([task_problem])[:MAX_FAILURES],
            )

        verdict = (
            f"`{run.command_line}` exited with code {run.returncode} in {self._workspace()}, so the "
            "suite failed."
        )
        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Tasks attempted this cycle:\n{task_context}\n\n"
            f"Command executed: {run.command_line}\n"
            f"Working directory: {self._workspace()}\n"
            f"Exit code: {run.returncode}\n\n"
            f"Test output:\n{_truncate(run.output, MAX_PROMPT_OUTPUT_CHARS)}\n\n"
            f"The command exited with a non-zero status, so the suite FAILED regardless of what you "
            "write. Describe what broke in `summary` and list the concrete failures in `failures`."
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        data = _load_json_object(response)
        summary = str((data or {}).get("summary") or "").strip() or verdict
        combined = _dedupe([*_as_str_list((data or {}).get("failures")), *failures])
        if not combined:
            combined = [summary]
        logger.info("Tests failed: %s", summary)
        # The model may enrich the prose, never the verdict: the exit code already said "failed".
        return TestResult(passed=False, summary=summary, failures=combined[:MAX_FAILURES])

    async def _run_tests(self) -> TestRun:
        """Try each known test runner until one exists; return its structured outcome.

        Only a missing binary (``FileNotFoundError``) falls through to the next candidate. A
        timeout is reported as a result, not retried with an unrelated runner.
        """
        cwd = self._workspace()
        timeout = self._timeout_seconds()
        for cmd in TEST_COMMANDS:
            logger.debug("Running test command %s in %s", " ".join(cmd), cwd)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            except FileNotFoundError:
                logger.debug("Test runner %r is not installed; trying the next candidate", cmd[0])
                continue
            stdout, stderr, timed_out = await _communicate(proc, timeout)
            output = _decode(stdout) + _decode(stderr)
            if timed_out:
                logger.warning("Test command %s timed out after %ss", " ".join(cmd), timeout)
                return TestRun(
                    command=cmd,
                    returncode=None,
                    output=output,
                    timed_out=True,
                    found=True,
                    timeout_seconds=timeout,
                )
            return TestRun(
                command=cmd,
                returncode=proc.returncode,
                output=output,
                timed_out=False,
                found=True,
                timeout_seconds=timeout,
            )
        logger.warning("No test runner found after trying %d candidates in %s", len(TEST_COMMANDS), cwd)
        return TestRun()

    def _task_context(self, tasks: list[TaskModel]) -> tuple[str, str | None]:
        """Describe the attempted tasks and flag the suspiciously-incomplete cases."""
        if not tasks:
            return "(no task context was supplied to the tester)", None
        lines: list[str] = []
        incomplete: list[str] = []
        for task in tasks:
            files = ", ".join(task.files) if task.files else "unspecified files"
            lines.append(f"- [{task.id}] status={task.status} files=[{files}] :: {task.description}")
            if task.status != "completed":
                incomplete.append(f"{task.id} ({task.status})")
        context = "\n".join(lines)
        if len(incomplete) == len(tasks):
            return context, "none of the planned tasks completed this cycle: " + ", ".join(incomplete)
        if incomplete:
            return context, "tasks that did not complete this cycle: " + ", ".join(incomplete)
        return context, None

    def _success_summary(self, run: TestRun) -> str:
        tail = _output_tail(run.output, MAX_SUMMARY_TAIL_LINES)
        summary = f"`{run.command_line}` exited 0 in {self._workspace()}; the suite passed."
        return f"{summary}\n{tail}" if tail else summary

    def _tail_fallback(self, run: TestRun) -> str:
        lines = [line.rstrip() for line in run.output.splitlines() if line.strip()]
        tail = _truncate("\n".join(lines[-MAX_FALLBACK_TAIL_LINES:]), MAX_FALLBACK_TAIL_CHARS)
        body = tail or "(the runner produced no output)"
        return (
            f"`{run.command_line}` exited with code {run.returncode} in {self._workspace()}.\n"
            f"Last output lines:\n{body}"
        )

    def _workspace(self) -> str:
        """Directory the suite must run in, falling back to the CWD for a stale config."""
        configured = getattr(self.client.settings, "workspace", None)
        if configured is not None:
            try:
                workspace = Path(configured)
            except TypeError:
                workspace = None
            if workspace is not None and workspace.is_dir():
                return str(workspace)
            logger.warning(
                "Configured workspace %r is not a usable directory; using %s instead",
                configured,
                Path.cwd(),
            )
        return str(Path.cwd())

    def _timeout_seconds(self) -> float:
        configured = getattr(
            self.client.settings,
            "test_timeout_seconds",
            DEFAULT_TEST_TIMEOUT_SECONDS,
        )
        try:
            timeout = float(configured)
        except (TypeError, ValueError):
            logger.warning(
                "test_timeout_seconds=%r is not a number; using %g",
                configured,
                DEFAULT_TEST_TIMEOUT_SECONDS,
            )
            return DEFAULT_TEST_TIMEOUT_SECONDS
        if timeout <= 0:
            logger.warning("test_timeout_seconds=%g is not positive; using %g", timeout, DEFAULT_TEST_TIMEOUT_SECONDS)
            return DEFAULT_TEST_TIMEOUT_SECONDS
        return timeout