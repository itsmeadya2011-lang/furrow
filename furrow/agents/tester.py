from __future__ import annotations

import asyncio
import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from furrow.agents.parsing import extract_json
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult, get_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Upper bound on a single test run before its process is killed.
TEST_TIMEOUT_SECONDS = 120

# Tried in order when no explicit test command is configured. The first candidate
# that can actually be spawned wins, even if its suite is red: a failing run must
# never be hidden by silently falling through to a different runner.
_TEST_CANDIDATES: tuple[str, ...] = (
    "pytest -q",
    "python -m pytest -q",
    "npm test -- --silent",
    "pnpm test -- --silent",
    "yarn test --silent",
    "cargo test -q",
    "go test ./...",
)

_ZERO_COUNT_RE = re.compile(r"\b0\s+(?:fail\w*|error\w*)\b")
_FAILURE_MARKERS = ("fail", "error", "exception", "traceback", "not pass", "cannot", "unable")
_PASS_MARKERS = ("pass", "green", "success")


@dataclass
class CommandResult:
    """Outcome of one test command invocation.

    ``found`` is False only when no runner could be started at all, which is a
    different situation from a runner that started and failed. ``returncode`` is
    the process exit status, or None when the process never ran.
    """

    command: str
    returncode: Optional[int]
    output: str
    found: bool
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        """True only when the command ran to completion with a zero exit code."""
        return self.found and not self.timed_out and self.returncode == 0

    def render(self) -> str:
        """Format the run for the model, exit code first so it cannot be missed."""
        if not self.found:
            return self.output
        header = f"$ {self.command}\nexit code: {self.returncode}"
        if self.timed_out:
            header += f"\ntimed out after {TEST_TIMEOUT_SECONDS}s and was killed"
        return f"{header}\n{self.output}"


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(settings=self.settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            result = await self._run_tests()
        except OSError as exc:
            # Last-resort guard, not the spawn-time case: _execute already turns
            # an unspawnable command into a "not found" result, but a process that
            # did start can still fail while it is collected (e.g. a timed-out
            # command that exits just before it is killed).
            return _failure(f"Failed to execute the test command: {exc}")

        if not result.found:
            message = "No test runner found; test results could not be verified."
            return TestResult(passed=False, summary=message, failures=[message])

        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n"
            f"Tasks that were supposed to be implemented:\n{_render_tasks(tasks)}\n\n"
            f"Test output:\n{result.render()}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        return self._combine(result, _verdict(response))

    async def _run_tests(self) -> CommandResult:
        """Run the configured test command, or auto-detect a runner.

        Auto-detection skips a candidate only when its binary cannot be spawned.
        The first candidate that starts is authoritative: its real exit code is
        reported even when the suite is red. A configured command is reported the
        same way when it cannot be started, so this never returns None.
        """
        if self.settings.test_command:
            # An explicit override is never second-guessed by auto-detection, so
            # an unspawnable command is reported rather than retried.
            command = self.settings.test_command
            result = await self._execute(command, shell=True)
            if result is None:
                return CommandResult(
                    command=command,
                    returncode=None,
                    output=(
                        f"No test runner found: the configured test command could "
                        f"not be executed: {command}"
                    ),
                    found=False,
                )
            return result
        for command in _TEST_CANDIDATES:
            result = await self._execute(command, shell=False)
            if result is None or result.timed_out:
                # Not installed, or it hung: fall through to the next runner.
                continue
            return result
        return CommandResult(
            command="", returncode=None, output="No test runner found.", found=False
        )

    async def _execute(self, command: str, *, shell: bool) -> CommandResult | None:
        """Run ``command`` once.

        Returns None when the process could not be spawned at all, so the caller
        can try the next candidate. Once a process exists, its real result is
        always returned.
        """
        try:
            if shell:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=self.settings.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *shlex.split(command),
                    cwd=self.settings.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
        except OSError:
            # The binary (or the working directory) is not usable. Everything
            # after a successful spawn is a genuine test result, not a fallback.
            return None
        return await self._collect(proc, command)

    async def _collect(self, proc: asyncio.subprocess.Process, command: str) -> CommandResult:
        """Wait for ``proc``, capturing its merged output and real exit code."""
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=TEST_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()  # reap the process; never leave it running or zombie
            return CommandResult(
                command=command,
                returncode=proc.returncode,
                output=f"Command exceeded the {TEST_TIMEOUT_SECONDS}s timeout.",
                found=True,
                timed_out=True,
            )
        return CommandResult(
            command=command,
            returncode=proc.returncode,
            output=stdout.decode(errors="replace") + stderr.decode(errors="replace"),
            found=True,
        )

    def _combine(self, result: CommandResult, verdict: TestResult) -> TestResult:
        """Combine the objective exit status with the model's verdict.

        The exit code is authoritative: a non-zero code or a timeout is never
        reported as a pass, whatever the model claims. The model's summary and
        failure list are kept, with the objective reason appended.
        """
        if result.succeeded:
            return verdict
        reason = self._failure_reason(result)
        summary = f"{verdict.summary}\n{reason}" if verdict.summary else reason
        failures = list(verdict.failures)
        if not any(reason in failure for failure in failures):
            failures.append(reason)
        return TestResult(passed=False, summary=summary, failures=failures)

    @staticmethod
    def _failure_reason(result: CommandResult) -> str:
        if result.timed_out:
            return f"Test command timed out after {TEST_TIMEOUT_SECONDS}s: {result.command}"
        return f"Test command exited with code {result.returncode}: {result.command}"


def _failure(message: str) -> TestResult:
    return TestResult(passed=False, summary=message, failures=[message])


def _verdict(response: str) -> TestResult:
    """Read the model's reply as a TestResult, with a conservative text fallback."""
    try:
        return TestResult(**extract_json(response))
    except (TypeError, ValueError):
        return TestResult(passed=_text_indicates_pass(response), summary=response, failures=[])


def _text_indicates_pass(response: str) -> bool:
    """Last-resort verdict for a reply that is not valid JSON.

    Zero counts such as "0 failed" are neutralised first, then any mention of
    failures, errors or uncertainty resolves to False. Only an unqualified
    statement that the tests passed yields True; anything else is a failure.
    """
    cleaned = _ZERO_COUNT_RE.sub(" ", response.lower())
    if any(marker in cleaned for marker in _FAILURE_MARKERS):
        return False
    return any(marker in cleaned for marker in _PASS_MARKERS)


def _render_tasks(tasks: list[TaskModel]) -> str:
    """Describe the planned tasks so the tester knows what should have been built."""
    if not tasks:
        return "  (no tasks were planned)"
    lines = []
    for task in tasks:
        status = getattr(task.status, "value", task.status)
        files = ", ".join(task.files) or "unspecified"
        dependencies = ", ".join(task.dependencies) or "none"
        lines.append(
            f"- id={task.id} [{status}]: {task.description}"
            f" | files: {files} | depends on: {dependencies}"
        )
    return "\n".join(lines)
