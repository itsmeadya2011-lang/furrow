from __future__ import annotations

import asyncio
import logging
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from furrow.agents._shared import as_str_list, as_text, truncate
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)

#: Used when ``Settings.test_timeout_seconds`` is missing, non-numeric or non-positive.
DEFAULT_TEST_TIMEOUT_SECONDS = 120
#: Seconds allowed to reap a killed runner before giving up on it.
REAP_TIMEOUT_SECONDS = 5
#: Summaries and failures are shown to the model and echoed to the user; keep them bounded.
MAX_SUMMARY_CHARS = 2000
MAX_FAILURE_CHARS = 500
MAX_FAILURES = 20

#: What a "this runner is not installed here" failure looks like. ``asyncio.CancelledError`` is
#: deliberately absent: cancellation must never be mistaken for a missing runner.
MISSING_RUNNER_ERRORS = (FileNotFoundError, NotADirectoryError, PermissionError)

#: Test runners in preference order; the first installed one wins. The interpreter running
#: furrow leads, because ``sys.executable -m pytest`` is guaranteed to match the environment
#: pytest is installed in, while a bare ``python`` or a PATH-resolved ``pytest`` may not.
TEST_COMMANDS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "-m", "pytest", "-q"),
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("npm", "test", "--", "--silent"),
    ("pnpm", "test", "--", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)

_TRUTHY = frozenset({"true", "t", "yes", "y", "pass", "passed", "ok", "ok!", "1", "on", "green"})
_FALSY = frozenset({"false", "f", "no", "n", "fail", "failed", "0", "off", "red"})
#: Substrings that mark a raw test-output line as worth feeding into the next planning goal.
FAILURE_HINTS = ("failed", "fail:", "failing", "error", "assertionerror", "traceback", "✗", "×")
#: Test output that says nothing either way, so it cannot be read as evidence of success.
INCONCLUSIVE_MARKERS = ("no test runner found", "timed out after")


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _as_bool(value: Any) -> bool | None:
    """Coerce a verdict's ``passed`` field, or ``None`` when it says nothing usable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUTHY:
            return True
        if lowered in _FALSY:
            return False
    return None


def _failure_lines(test_output: str) -> list[str]:
    """Pull failure-looking lines out of raw test output, for the next planning goal."""
    lines: list[str] = []
    for raw_line in test_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not any(hint in line.lower() for hint in FAILURE_HINTS):
            continue
        lines.append(truncate(line, MAX_FAILURE_CHARS))
        if len(lines) >= MAX_FAILURES:
            break
    return lines


def _fallback_summary(test_output: str) -> str:
    output = test_output.strip()
    if not output:
        return "The tester returned no summary."
    return truncate(output, MAX_SUMMARY_CHARS)


def _ran_cleanly(test_output: str) -> bool:
    """Whether the raw run is evidence of success: no failure lines and a conclusive output.

    Used only when the model's ``passed`` field says nothing usable. The substring heuristic is
    useless there, because a parsed verdict always contains the key ``"passed"`` itself.
    """
    if _failure_lines(test_output):
        return False
    lowered = test_output.strip().lower()
    if not lowered:
        return False
    return not any(marker in lowered for marker in INCONCLUSIVE_MARKERS)


def _heuristic_result(response: str, test_output: str, reason: str | None = None) -> TestResult:
    """Last-resort verdict from the ``"passed" in response`` heuristic.

    A failed verdict must still carry ``failures``: the orchestrator feeds that list into the
    next planning goal, so an empty one would silently lose the only actionable detail.
    """
    passed = "passed" in response.lower()
    summary = as_text(response)
    if not summary:
        summary = (
            f"The tester response was unusable ({reason})."
            if reason
            else "The tester returned an empty response."
        )
    summary = truncate(summary, MAX_SUMMARY_CHARS)
    failures = _failure_lines(test_output) if not passed else []
    if not passed and not failures:
        failures = [truncate(summary, MAX_FAILURE_CHARS)]
    return TestResult(passed=passed, summary=summary, failures=failures)


def _verdict_fields(data: dict[str, Any], test_output: str) -> dict[str, Any]:
    """Normalise a parsed verdict into exact ``TestResult`` field types."""
    passed = _as_bool(data.get("passed"))
    if passed is None:
        # ``passed`` said nothing usable, so decide from the run itself: never report success
        # without evidence, and never on a run that never happened.
        passed = _ran_cleanly(test_output)
    summary = as_text(data.get("summary")) or _fallback_summary(test_output)
    failures = as_str_list(data.get("failures"))
    if not passed and not failures:
        failures = _failure_lines(test_output) or [truncate(summary, MAX_FAILURE_CHARS)]
    return {"passed": passed, "summary": summary, "failures": failures}


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.test_commands: tuple[tuple[str, ...], ...] = TEST_COMMANDS

    @property
    def settings(self) -> Settings | None:
        """The settings in force.

        Read off the client so ``TesterAgent(client=...)`` -- how the orchestrator builds it --
        honours the client's configuration rather than a module-level default.
        """
        return getattr(self.client, "settings", None)

    @property
    def test_timeout_seconds(self) -> float:
        """Seconds a single test run may take, from settings with a sane floor."""
        raw = getattr(self.settings, "test_timeout_seconds", DEFAULT_TEST_TIMEOUT_SECONDS)
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric test_timeout_seconds=%r.", raw)
            return float(DEFAULT_TEST_TIMEOUT_SECONDS)
        if timeout <= 0:
            logger.warning(
                "Clamping non-positive test_timeout_seconds=%r to %d.",
                raw,
                DEFAULT_TEST_TIMEOUT_SECONDS,
            )
            return float(DEFAULT_TEST_TIMEOUT_SECONDS)
        return timeout

    @property
    def workspace(self) -> str | None:
        """The configured workspace, used as the subprocess ``cwd``.

        ``None`` (keep the process cwd) when unset or not a directory: a bad ``cwd`` would make
        every candidate look like a missing runner.
        """
        raw = getattr(self.settings, "workspace", None)
        if raw is None:
            return None
        path = Path(str(raw))
        if not path.is_dir():
            logger.warning("Configured workspace %s is not a directory; not changing cwd.", path)
            return None
        return str(path)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_output = await self._run_tests()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Test run raised %s: %s", type(exc).__name__, exc)
            message = f"Test run raised {type(exc).__name__}: {exc}"
            return TestResult(passed=False, summary=message, failures=[message])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        return self._verdict(response, test_output)

    @staticmethod
    def _verdict(response: str, test_output: str) -> TestResult:
        """Turn the tester model's answer into a :class:`TestResult`.

        A malformed verdict must never kill a cycle: anything unusable degrades to the
        ``"passed" in response.lower()`` heuristic, which still carries a failure list.
        """
        try:
            data = extract_json(response)
        except (ValueError, TypeError) as exc:
            logger.warning("Tester response is not usable JSON (%s); using the heuristic.", exc)
            return _heuristic_result(response, test_output, f"unparseable verdict: {exc}")
        if not isinstance(data, dict):
            logger.warning(
                "Tester response parsed as %s, not an object; using the heuristic.",
                type(data).__name__,
            )
            return _heuristic_result(response, test_output, "verdict was not a JSON object")
        try:
            return TestResult(**_verdict_fields(data, test_output))
        except ValidationError as exc:
            logger.warning("Tester verdict rejected by TestResult (%s); using the heuristic.", exc)
            return _heuristic_result(response, test_output, f"invalid verdict: {exc}")

    async def _run_tests(self) -> str:
        """Run the first installed test runner and return its combined output.

        Three outcomes, kept distinct on purpose:

        * runner not installed -> try the next candidate;
        * runner timed out -> kill and reap it, report the timeout, and stop: the suite really
          did start, so running another candidate would only double the wait;
        * runner finished -> return its output, non-zero exit code included, because that is
          exactly what a failing suite looks like.
        """
        timeout = self.test_timeout_seconds
        cwd = self.workspace
        for cmd in self.test_commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except MISSING_RUNNER_ERRORS:
                logger.debug(
                    "Test runner %r is not available here; trying the next candidate.", cmd[0]
                )
                continue
            return await self._collect(proc, cmd, timeout, cwd)
        return "No test runner found."

    async def _collect(
        self,
        proc: asyncio.subprocess.Process,
        cmd: tuple[str, ...],
        timeout: float,
        cwd: str | None,
    ) -> str:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._kill(proc)
            logger.warning(
                "Test runner %r exceeded %.1fs in %s; killed it.",
                cmd[0],
                timeout,
                cwd or ".",
            )
            return (
                f"Tests timed out after {timeout:g}s running `{shlex.join(cmd)}`; "
                "the test runner was killed."
            )
        except BaseException:
            # Includes CancelledError: never leave a runner behind on cancellation.
            await self._kill(proc)
            raise
        output = _decode(stdout) + _decode(stderr)
        if proc.returncode and not output.strip():
            return f"`{shlex.join(cmd)}` exited with code {proc.returncode} and printed nothing."
        return output

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process) -> None:
        """Kill ``proc`` and reap it, so no child outlives the test run."""
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:  # already exited between the check and the signal
                logger.debug("Test runner %s was already gone.", proc.pid)
        try:
            await asyncio.wait_for(proc.wait(), timeout=REAP_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("Test runner %s did not exit after being killed.", proc.pid)
