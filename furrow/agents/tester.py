from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.core.json_utils import extract_json
from furrow.llm import LLMClient
from furrow.logging_config import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger(__name__)

# Markers emitted by _format_result; the strongest signal the LLM (and the
# fallback heuristic) get about whether the run actually passed.
_RESULT_RE = re.compile(r"^RESULT:\s*(PASSED|FAILED)\b", re.MULTILINE | re.IGNORECASE)

# "0 failed" / "0 errors" only mean success if zero-failure is checked before
# the substring checks below, since "0 failed" contains "fail".
_ZERO_PROBLEM_RE = re.compile(r"\b0\s+(?:failed|failures?|errors?)\b")

_CLEAN_SUCCESS = (
    "all tests passed",
    "all tests successful",
    "no failures",
    "no failing tests",
    "no errors",
    "everything passed",
)

_FAILURE = (
    "fail",
    "error",
    "exception",
    "traceback",
    "fatal",
    "panic",
    "assert",
    "did not pass",
    "not pass",
    "not passed",
    "not run",
    "not found",
    "not installed",
    "not executed",
    "were not run",
    "no tests ran",
    "timed out",
    "timeout",
    "denied",
    "refused",
    "unable",
    "cannot",
)

_PASS_HINT = ("pass", "ok", "success", "green")

MISSING_RUNNER = (
    "No test runner found: none of the candidate commands "
    "(pytest, python -m pytest, npm, pnpm, yarn, cargo, go) could be executed, "
    "so tests were NOT run and this must not be reported as a passing run."
)


def _snippet(text: str, limit: int = 500) -> str:
    """Return a truncated copy of ``text`` for inclusion in summaries."""
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def _text_indicates_pass(text: str) -> bool:
    """Conservative verdict for unparseable tester output.

    Plain text counts as PASSED only when it clearly says so and says nothing
    about problems; anything ambiguous is a failure, because a false pass makes
    the orchestrator stop asking for fixes.
    """
    match = _RESULT_RE.search(text)
    if match:
        return match.group(1).upper() == "PASSED"

    lowered = text.lower()
    if _ZERO_PROBLEM_RE.search(lowered):
        return True
    if any(token in lowered for token in _CLEAN_SUCCESS):
        return True
    if any(token in lowered for token in _FAILURE):
        return False
    return any(token in lowered for token in _PASS_HINT)


def _format_result(cmd: list[str], returncode: int, output: str, cwd: str) -> str:
    verdict = "PASSED" if returncode == 0 else "FAILED"
    header = (
        f"Test command: {' '.join(cmd)}\n"
        f"Working directory: {cwd}\n"
        f"Exit code: {returncode} ({verdict})\n"
        f"RESULT: {verdict}"
    )
    return f"{header}\n\n{output}" if output else header


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the whole process group started for ``proc`` (POSIX), else the child."""
    getpgid = getattr(os, "getpgid", None)
    killpg = getattr(os, "killpg", None)
    if getpgid is not None and killpg is not None:
        try:
            pgid = getpgid(proc.pid)
        except OSError:
            pgid = None
        if pgid is not None:
            with contextlib.suppress(OSError):
                killpg(pgid, signal.SIGKILL)
                return
    with contextlib.suppress(OSError, ProcessLookupError):
        proc.kill()


class TesterAgent:
    # Not a pytest test class: pytest collects any imported class named Test* and
    # warns about this one because it has an __init__. Keep the marker so the
    # warning stays gone.
    __test__ = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the test suite and ask the model to judge the output against ``goal``.

        Every error raised by the test run itself becomes a reported failure,
        never a silent skip, so the ``except Exception`` below is deliberate
        policy rather than a blanket catch (BLE/S110 are intentionally not
        selected in pyproject for exactly this reason).
        """
        try:
            test_output = await self._run_tests()
        except Exception as e:
            # A broken test run is a failure to report, never a silent skip.
            message = f"{type(e).__name__}: {e}"
            log.warning("test run raised; reported as a failure", error=message)
            return TestResult(passed=False, summary=message, failures=[message])

        log.info("tester judging run", goal_chars=len(goal), tasks=len(tasks))

        # The planned tasks are context for the verdict: the tester needs to know
        # which ids were meant to be implemented to attribute a failure.
        context = "\n".join(f"- {task.id}: {task.description}" for task in tasks)
        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n"
            f"Planned tasks:\n{context}\n\n"
            f"Test output:\n{test_output}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            # Tolerates ```json fences, leading prose and trailing commentary.
            data = extract_json(response)
            if not isinstance(data, dict):
                raise TypeError(f"expected a JSON object, got {type(data).__name__}")
            return TestResult(**data)
        except (ValueError, TypeError):
            passed = _text_indicates_pass(response)
            summary = "Tester LLM returned unparseable output; "
            summary += (
                "output analysis says the tests passed."
                if passed
                else "output analysis is ambiguous or negative, so it is treated as FAILED."
            )
            return TestResult(
                passed=passed,
                summary=summary,
                failures=[] if passed else [response],
            )

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]

        workspace = Path(self.client.settings.workspace)
        if not workspace.is_dir():
            # Reported as an error rather than "no runner found", which would lie.
            raise FileNotFoundError(f"Workspace directory does not exist: {workspace}")
        cwd = str(workspace)
        timeout = self.client.settings.test_timeout

        for cmd in candidates:
            try:
                output = await self._run_candidate(cmd, cwd, timeout)
            except (FileNotFoundError, NotADirectoryError):
                # Runner is not installed here; try the next candidate.
                continue
            log.info("test runner selected", command=cmd, cwd=cwd, timeout=timeout)
            return output
        log.warning("no test runner found; tests were not run", candidates=len(candidates))
        return MISSING_RUNNER

    async def _run_candidate(self, cmd: list[str], cwd: str, timeout: int) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            # Own process group, so a timeout can kill the whole tree.
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            _kill_process_group(proc)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5)
            # A hang is a failure with a known cause, not a missing test runner.
            return (
                f"Test command: {' '.join(cmd)}\n"
                f"Working directory: {cwd}\n"
                f"RESULT: FAILED\n"
                f"Timed out after {timeout}s and the process group was killed. "
                f"No output was produced."
            )
        output = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()
        # communicate() has returned, so the process has been reaped and returncode
        # is set; the stubs still type it as int | None. The fallback keeps the exit
        # code the authority on pass/fail and renders as FAILED if it is ever hit.
        returncode = proc.returncode if proc.returncode is not None else -1
        return _format_result(cmd, returncode, output, cwd)
