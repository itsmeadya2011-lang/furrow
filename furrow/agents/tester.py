from __future__ import annotations

import asyncio
import contextlib
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

#: Candidate test runners, tried in order when ``settings.test_command`` is unset.
TEST_COMMANDS: Final[tuple[tuple[str, ...], ...]] = (
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("npm", "test", "--", "--silent"),
    ("pnpm", "test", "--", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)
#: Hard cap on the number of failures reported in a ``TestResult``.
MAX_FAILURES: Final[int] = 20
#: How much runner output reaches the LLM; the tail is kept because pytest prints
#: its failure summary last.
MAX_PROMPT_OUTPUT: Final[int] = 20_000
#: A workspace with no test runner is not a failure: nothing was broken.
NO_RUNNER_SUMMARY: Final[str] = "No test runner found; nothing to verify."

_FAILED_RE = re.compile(r"^FAILED\s+(?P<test>\S+)")
_ASSERTION_RE = re.compile(r"^E\s+(?P<assertion>\S.*?)\s*$")


@dataclass
class CommandResult:
    """Outcome of a single test-runner invocation.

    ``ran`` is False when the executable does not exist, so ``run`` can tell "no
    runner available" apart from "the runner ran and the tests failed". ``exit_code``
    is None only for a run that timed out and was killed.
    """

    ran: bool
    exit_code: int | None
    output: str


def extract_failures(output: str, limit: int = MAX_FAILURES) -> list[str]:
    """Return de-duplicated, capped failure descriptions pulled out of raw output.

    Recognises pytest's ``FAILED <path>::<name>`` summary lines and its ``E   <expr>``
    assertion lines, keeping the order in which they appear.
    """
    failures: list[str] = []
    seen: set[str] = set()
    for line in output.splitlines():
        failed = _FAILED_RE.match(line)
        if failed:
            entry = f"FAILED {failed['test'].split(' - ')[0]}"
        else:
            assertion = _ASSERTION_RE.match(line)
            if not assertion:
                continue
            entry = assertion["assertion"]
        if entry in seen:
            continue
        seen.add(entry)
        failures.append(entry)
        if len(failures) >= max(0, limit):
            break
    return failures


def _format_output(
    cmd: Sequence[str],
    exit_code: int | None,
    stdout: str = "",
    stderr: str = "",
    note: str = "",
) -> str:
    """Render the command line, exit code and both output streams as one block."""
    header = f"$ {shlex.join(cmd)}\nexit code: {'timeout' if exit_code is None else exit_code}"
    if note:
        header = f"{header}\n{note}"
    sections = [header]
    if stdout.strip():
        sections.append(f"--- stdout ---\n{stdout.rstrip()}")
    if stderr.strip():
        sections.append(f"--- stderr ---\n{stderr.rstrip()}")
    return "\n".join(sections)


def _local_summary(result: CommandResult) -> str:
    """Describe the local verdict without involving the model."""
    if result.exit_code is None:
        return "Test runner timed out and was killed."
    return f"Test runner exited with code {result.exit_code}."


def _clip_output(output: str, limit: int = MAX_PROMPT_OUTPUT) -> str:
    """Keep the tail of ``output`` so the pytest failure summary survives truncation."""
    if len(output) <= limit:
        return output
    return f"...(truncated to the last {limit} characters)...\n{output[-limit:]}"


async def _reap(process: asyncio.subprocess.Process) -> None:
    """Kill a subprocess and await it so the child is reaped rather than leaked."""
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):  # already gone is fine
            process.kill()
    await process.wait()


class TesterAgent:
    """Runs the project's test runner locally, then asks the LLM to explain it."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings: Settings = settings if settings is not None else self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Return the test verdict; ``passed`` comes from the runner's exit code.

        The LLM only summarises: its verdict is ignored, and if it is unavailable the
        locally extracted failures and a local summary are returned instead.
        """
        try:
            result = await self._run_tests()
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed result
            detail = f"{type(exc).__name__}: {exc}"
            return TestResult(
                passed=False, summary=f"Could not run the test runner: {detail}", failures=[detail]
            )
        if not result.ran:
            return TestResult(passed=True, summary=NO_RUNNER_SUMMARY)
        passed = result.exit_code == 0
        failures = extract_failures(result.output)
        try:
            data = await self.client.complete_json(
                self._build_prompt(goal, tasks, result),
                model=self.client.settings.tester_model,
            )
        except Exception:  # noqa: BLE001 - the local verdict stands on its own
            return TestResult(passed=passed, summary=_local_summary(result), failures=failures)
        return self._report(data, result, passed, failures)

    async def _run_tests(self) -> CommandResult:
        """Run the configured test command, or the first available candidate runner."""
        cwd = str(self.settings.workspace)
        # The `stat` behind `is_dir` blocks, so it goes to a worker thread.
        if not await asyncio.to_thread(Path(cwd).is_dir):
            raise FileNotFoundError(f"Workspace does not exist: {cwd}")
        configured = (self.settings.test_command or "").strip()
        commands: tuple[tuple[str, ...], ...] = (
            (tuple(shlex.split(configured)),) if configured else TEST_COMMANDS
        )
        timed_out: CommandResult | None = None
        for cmd in commands:
            result = await self._run_command(cmd, cwd)
            if result.ran and result.exit_code is not None:
                return result
            if result.ran:
                timed_out = result  # keep the timeout reason if nothing else launches
        if timed_out is not None:
            return timed_out
        return CommandResult(ran=False, exit_code=None, output=NO_RUNNER_SUMMARY)

    async def _run_command(self, cmd: tuple[str, ...], cwd: str) -> CommandResult:
        """Run one candidate; ``ran`` is False when the executable does not exist."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return CommandResult(ran=False, exit_code=None, output="")
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.settings.test_timeout
            )
        except asyncio.TimeoutError:
            await _reap(process)
            note = f"timed out after {self.settings.test_timeout:g}s; process killed"
            return CommandResult(
                ran=True,
                exit_code=None,
                output=_format_output(cmd, None, note=note),
            )
        return CommandResult(
            ran=True,
            exit_code=process.returncode,
            output=_format_output(cmd, process.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")),
        )

    @staticmethod
    def _build_prompt(goal: str, tasks: list[TaskModel], result: CommandResult) -> str:
        """Assemble the tester prompt from the goal, the plan and the runner output."""
        sections = [TESTER_PROMPT, f"Goal: {goal}"]
        lines = [f"- [{task.id}] {task.description}" for task in tasks if task.description.strip()]
        if lines:
            sections.append("Tasks:\n" + "\n".join(lines))
        sections.append(f"Test output:\n{_clip_output(result.output)}")
        sections.append(
            f"The test runner exited with code {result.exit_code}; that exit code is "
            "authoritative for the passed verdict, so do not contradict it."
        )
        return "\n\n".join(sections)

    @staticmethod
    def _report(
        data: dict[str, Any],
        result: CommandResult,
        passed: bool,
        failures: list[str],
    ) -> TestResult:
        """Keep the local verdict, adopting the model's summary and failures."""
        summary = data.get("summary")
        summary = summary.strip() if isinstance(summary, str) else ""
        reported = data.get("failures")
        merged = (
            [str(item).strip() for item in reported if str(item).strip()]
            if isinstance(reported, list)
            else []
        )
        verdict = data.get("passed")
        if isinstance(verdict, bool) and verdict != passed:
            note = (
                f"Local verdict from exit code {result.exit_code}: "
                f"{'passed' if passed else 'failed'}"
            )
            summary = f"{summary} ({note})" if summary else note
        return TestResult(
            passed=passed,
            summary=summary or _local_summary(result),
            failures=(merged or failures)[:MAX_FAILURES],
        )