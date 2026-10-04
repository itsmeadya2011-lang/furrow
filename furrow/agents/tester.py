from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from collections.abc import Sequence

    from furrow.config import Settings

logger = logging.getLogger(__name__)

# Hard cap on captured test output. We keep the TAIL because that is where
# summaries, failures and tracebacks live.
MAX_OUTPUT_CHARS = 20_000
# How long we wait for a killed test process to actually go away before
# giving up on reaping it.
REAP_TIMEOUT = 5.0
DEFAULT_TEST_TIMEOUT = 120


@dataclass
class TestRun:
    """Result of executing a test command once.

    ``available`` is False when nothing actually ran (no runner installed / no
    test command configured), which is *not* the same thing as tests failing.
    """

    output: str
    exit_code: int | None = None
    runner: str | None = None  # e.g. "pytest -q"; None if nothing ran
    timed_out: bool = False
    available: bool = False  # True if a runner actually executed

    @property
    def ok(self) -> bool:
        """True only when a runner ran and exited 0."""
        return self.available and not self.timed_out and self.exit_code == 0


# Ordered candidates: the first one whose executable exists AND actually runs
# wins. We never fall through on failure -- a failing suite is a result.
CANDIDATE_RUNNERS: tuple[tuple[str, ...], ...] = (
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("npm", "test", "--", "--silent"),
    ("pnpm", "test", "--", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)

# `python -m pytest` "runs" successfully on any box with python installed, even
# without pytest. Treat that as "runner not installed" rather than a failure.
_MISSING_PYTEST = re.compile(r"No module named ['\"]?pytest", re.IGNORECASE)

_FAILURE_LINE = re.compile(
    r"^(?:FAILED|ERROR|FAIL)\b"
    r"|^E\s+\S"
    r"|\bTraceback \(most recent call last\)"
    r"|\bAssertionError\b"
    r"|^File \""
    r"|\berror(?:\[|:)"
    r"|\bpanic(?:ked)?:",
    re.IGNORECASE,
)

_PASS_WORD = re.compile(
    r"\bpass(?:ed|es|ing)?\b|\bsuccess(?:ful|fully)?\b|\bgreen\b", re.IGNORECASE
)
_FAIL_WORD = re.compile(
    r"\bfail(?:ed|ing|ure|ures)?\b|\berrors?\b|\bbroken\b|\btraceback\b|\bexception\b",
    re.IGNORECASE,
)
_NEGATIONS = ("not ", "n't", "no ", "never ", "cannot ", "unsuccessful", "didn't", "doesn't")


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the tail of *text*; failures live at the end of a test log."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"... [{dropped} earlier characters truncated; showing the tail] ...\n{text[-limit:]}"


def _extract_failures(output: str, limit: int = 20) -> list[str]:
    """Best-effort extraction of actionable failure lines from noisy output."""
    failures: list[str] = []
    for raw in output.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if _FAILURE_LINE.search(line):
            stripped = line.strip()
            if stripped and stripped not in failures:
                failures.append(stripped)
    if failures:
        return failures[:limit]
    # Nothing recognisable -- fall back to the tail so the caller still gets
    # something concrete to act on instead of a giant blob.
    tail = [line.strip() for line in output.splitlines() if line.strip()][-5:]
    return tail


def _looks_like_pass(text: str) -> bool:
    """True only if an un-negated pass word appears ("tests did not pass" -> False)."""
    for match in _PASS_WORD.finditer(text):
        prefix = text[max(0, match.start() - 40) : match.start()].lower()
        if any(cue in prefix for cue in _NEGATIONS):
            continue
        return True
    return False


def _status_of(task: TaskModel) -> str:
    status = getattr(task, "status", "unknown")
    return str(getattr(status, "value", status))


def _clip(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _summarize_tasks(tasks: list[TaskModel]) -> str:
    """Compact id + status + description list so the model can judge the work."""
    if not tasks:
        return "(no tasks were planned)"
    lines = []
    for task in tasks:
        lines.append(f"{task.id}. [{_status_of(task)}] {_clip(task.description)}")
    return "\n".join(lines)


def _json_candidates(response: str) -> list[str]:
    """Plausible JSON payloads inside a (possibly chatty) model response."""
    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(.+?)```", response, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    stripped = response.strip()
    candidates.append(stripped)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    return candidates


def _parse_verdict(response: str) -> TestResult | None:
    """Parse the model verdict, or None if the response has no usable JSON."""
    for candidate in _json_candidates(response):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            return None
        raw_passed = data.get("passed")
        if isinstance(raw_passed, str):
            passed = raw_passed.strip().lower() in {"true", "yes", "1"}
        else:
            passed = raw_passed is True
        summary = str(data.get("summary") or "").strip()
        raw_failures = data.get("failures")
        if isinstance(raw_failures, str):
            failures = [raw_failures.strip()] if raw_failures.strip() else []
        elif isinstance(raw_failures, list):
            failures = [str(item).strip() for item in raw_failures if str(item).strip()]
        else:
            failures = []
        return TestResult(passed=passed, summary=summary, failures=failures)
    return None


def _fallback_verdict(response: str) -> TestResult:
    """Conservative verdict for unparseable output: failure unless clearly OK."""
    text = response.strip()
    passed = _looks_like_pass(text) and not _FAIL_WORD.search(text)
    summary = _truncate(text, 2_000) or "Tester returned an empty, unparseable response."
    if not passed:
        summary = f"{summary}\n(No parsable verdict in the tester response; treating as FAILURE.)"
    return TestResult(passed=passed, summary=summary, failures=[])


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the tests and combine the exit code with the model's verdict.

        The process exit code is authoritative: a non-zero exit is a failure
        even if the model claims success.
        """
        try:
            run = await self.run_command()
        except Exception as exc:  # never let a broken runner crash the loop
            logger.warning("could not run tests: %s", exc, exc_info=True)
            return TestResult(
                passed=False,
                summary=f"Could not execute tests: {exc}",
                failures=[f"Could not execute tests: {exc}"],
            )

        if not run.available:
            summary = (
                "No test runner was detected, so the tests were not executed and the "
                "goal could not be verified. This is not evidence that the code is broken."
            )
            if run.output:
                summary = f"{summary} {run.output}"
            failure = "No test runner available; install one or configure a test command."
            return TestResult(passed=False, summary=summary, failures=[failure])

        signal_failures = _extract_failures(run.output)

        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Planned tasks (work that was supposed to be done):\n{_summarize_tasks(tasks)}\n\n"
            "Test run facts (the exit code is authoritative -- if it is non-zero the "
            "tests failed even if the log looks clean):\n"
            f"- command: {run.runner}\n"
            f"- exit code: {run.exit_code}\n"
            f"- timed out: {'yes' if run.timed_out else 'no'}\n"
            f"- output may be truncated to the last {MAX_OUTPUT_CHARS} characters\n\n"
            f"Test output:\n{run.output}\n"
        )

        try:
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        except Exception as exc:
            logger.warning("tester LLM call failed (%s); using exit-code verdict only", exc)
            return self._combine(run, TestResult(passed=run.ok, summary="", failures=[]))

        verdict = _parse_verdict(response)
        if verdict is None:
            logger.warning("tester response was not valid JSON; using conservative fallback")
            verdict = _fallback_verdict(response)
        return self._combine(run, verdict, signal_failures)

    async def run_command(self) -> TestRun:
        """Execute the configured or auto-detected test command exactly once."""
        configured = getattr(self.settings, "test_command", None)
        if configured:
            return await self._run_configured(configured)

        tried: list[str] = []
        for cmd in CANDIDATE_RUNNERS:
            label = " ".join(cmd)
            tried.append(label)
            executable = shutil.which(cmd[0])
            if executable is None:
                logger.debug("test runner %r not on PATH; skipping", cmd[0])
                continue
            try:
                run = await self._execute([executable, *cmd[1:]], label)
            except (FileNotFoundError, NotADirectoryError) as exc:
                logger.debug("test runner %r is not installed: %s", cmd[0], exc)
                continue
            except Exception as exc:
                logger.warning("test runner %r could not be executed: %s", label, exc)
                continue
            if run.available and _MISSING_PYTEST.search(run.output):
                logger.debug("%s exited non-zero because pytest is not importable", label)
                continue
            if run.available:
                return run
            logger.debug("test runner %r was not usable: %s", label, run.output)

        return TestRun(
            output=f"No test runner found (tried: {'; '.join(tried)}).",
            runner=None,
            available=False,
        )

    async def _run_configured(self, configured: str | Sequence[object]) -> TestRun:
        """Run an explicitly configured test command instead of auto-detection.

        *configured* is ``Settings.test_command`` -- a shell-style string that is
        split with :func:`shlex.split` -- or, when a caller has already split it, an
        arbitrary sequence of argv parts, each stringified before use.

        Invariant: ``TestRun.runner`` names the command that actually executed, so
        it is only ever set by :meth:`_build_run`. Every "nothing ran" branch
        below leaves it ``None`` and names the attempted command in ``output``,
        which is what ``run()`` surfaces to the user.
        """
        if isinstance(configured, str):
            try:
                parts = shlex.split(configured)
            except ValueError as exc:
                logger.warning("could not parse test_command %r: %s", configured, exc)
                return TestRun(
                    output=f"Configured test command is not runnable: {configured} ({exc})."
                )
        else:
            parts = [str(part) for part in configured]
        if not parts:
            logger.warning("test_command is empty; falling back to auto-detection")
            return await self.run_command()

        label = " ".join(parts)
        executable = shutil.which(parts[0])
        if executable is None:
            logger.warning("configured test command %r is not installed", label)
            return TestRun(output=f"Configured test command is not available: {label}.")
        try:
            return await self._execute([executable, *parts[1:]], label)
        except (FileNotFoundError, NotADirectoryError) as exc:
            logger.warning("configured test command %r is not installed: %s", label, exc)
            return TestRun(output=f"Configured test command is not available: {label}.")
        except Exception as exc:
            logger.warning("configured test command %r could not be executed: %s", label, exc)
            return TestRun(output=f"Failed to execute configured test command {label}: {exc}")

    async def _execute(self, cmd: list[str], label: str) -> TestRun:
        """Run *cmd* once; never falls through to another command."""
        cwd = self._workspace()
        timeout = self._timeout()
        logger.debug("running tests: %s (cwd=%s, timeout=%ss)", label, cwd, timeout)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # NOTE: asyncio.wait does not cancel the task on timeout, so we can kill
        # the process and still drain whatever partial output it produced.
        comm = asyncio.create_task(proc.communicate())
        done, _pending = await asyncio.wait({comm}, timeout=timeout)

        if done:
            try:
                stdout, stderr = comm.result()
            except Exception as exc:
                logger.warning("could not read output of %s: %s", label, exc)
                stdout, stderr = b"", b""
            return self._build_run(proc, label, stdout, stderr, timed_out=False)

        logger.warning(
            "test runner %s timed out after %ss; killing pid %s", label, timeout, proc.pid
        )
        proc.kill()
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(comm), timeout=REAP_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("test runner %s did not exit %ss after kill", label, REAP_TIMEOUT)
            proc.kill()
            comm.cancel()
            stdout, stderr = b"", b""
            await self._reap(proc)
        except Exception as exc:
            logger.warning("error while draining output of %s: %s", label, exc)
            stdout, stderr = b"", b""
            await self._reap(proc)
        return self._build_run(proc, label, stdout, stderr, timed_out=True)

    @staticmethod
    def _build_run(
        proc: asyncio.subprocess.Process,
        label: str,
        stdout: bytes,
        stderr: bytes,
        timed_out: bool,
    ) -> TestRun:
        text = stdout.decode(errors="replace") + stderr.decode(errors="replace")
        if timed_out:
            text = f"[TIMED OUT -- partial output follows]\n{text}"
        exit_code = proc.returncode
        if exit_code is None:
            logger.warning("test runner %s has no exit code yet", label)
        logger.debug("test runner %s exited with %s", label, exit_code)
        return TestRun(
            output=_truncate(text),
            exit_code=exit_code,
            runner=label,
            timed_out=timed_out,
            available=True,
        )

    @staticmethod
    async def _reap(proc: asyncio.subprocess.Process) -> None:
        """Make sure a killed child is waited on, so no zombie is left behind."""
        try:
            await asyncio.wait_for(proc.wait(), timeout=REAP_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("could not reap test process pid=%s", proc.pid)
        except Exception as exc:
            logger.warning("error reaping test process pid=%s: %s", proc.pid, exc)

    def _combine(
        self,
        run: TestRun,
        verdict: TestResult,
        signal_failures: list[str] | None = None,
    ) -> TestResult:
        """Combine the authoritative exit code with the model's opinion."""
        failures = list(verdict.failures)
        summary = verdict.summary.strip()

        if run.exit_code != 0 or run.timed_out:
            # The process itself decides: a non-zero exit / timeout is a failure.
            reason = self._failure_reason(run)
            logger.info("tests FAILED (%s)", reason)
            return TestResult(
                passed=False,
                summary=f"{reason}. {summary}" if summary else reason,
                failures=failures or list(signal_failures or []) or [reason],
            )

        # Exit code 0: trust the model unless it reported an explicit failure.
        if verdict.passed:
            logger.info("tests passed (exit code 0)")
            return TestResult(
                passed=True,
                summary=summary or f"{run.runner} exited 0.",
                failures=failures,
            )
        fallback_failures = failures or list(signal_failures or [])
        return TestResult(
            passed=False,
            summary=summary or f"{run.runner} exited 0 but the tester reported problems.",
            failures=fallback_failures or ["Reported as failing by the tester."],
        )

    def _failure_reason(self, run: TestRun) -> str:
        name = run.runner or "test command"
        if run.timed_out:
            return (
                f"{name} timed out after {self._timeout():g}s and was killed; the results "
                "are incomplete and were treated as a failure"
            )
        return f"{name} exited with code {run.exit_code}"

    def _timeout(self) -> float:
        value = getattr(self.settings, "test_timeout", DEFAULT_TEST_TIMEOUT)
        try:
            value = float(value)
        except (TypeError, ValueError):
            logger.warning("invalid test_timeout %r; using %s", value, DEFAULT_TEST_TIMEOUT)
            return float(DEFAULT_TEST_TIMEOUT)
        if value <= 0:
            logger.warning("non-positive test_timeout %r; using %s", value, DEFAULT_TEST_TIMEOUT)
            return float(DEFAULT_TEST_TIMEOUT)
        return value

    def _workspace(self) -> Path:
        """Configured workspace, resolved; falls back to the process CWD."""
        raw = getattr(self.settings, "workspace", None) or Path.cwd()
        try:
            workspace = Path(raw).expanduser().resolve()
        except OSError as exc:
            logger.warning("could not resolve workspace %s (%s); using %s", raw, exc, Path.cwd())
            return Path.cwd()
        if not workspace.is_dir():
            logger.warning("workspace %s is not a directory; using %s", workspace, Path.cwd())
            return Path.cwd()
        return workspace

    async def _run_tests(self) -> str:
        """Backwards-compatible wrapper: the raw captured output only."""
        return (await self.run_command()).output
