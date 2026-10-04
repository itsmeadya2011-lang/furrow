"""Tester agent: runs the workspace test suite and judges the verdict.

Two independent signals are collected. The process is the hard signal: a
non-zero exit status, a timeout or a missing binary is recorded in
:class:`CommandResult` and can never be reported as a pass. The model is the
soft signal: it explains *why* a run failed, which is what the orchestrator
turns into the next cycle's goal. When the two disagree, the exit status wins
and the override is stated in the summary rather than hidden.

Example:
    ```python
    tester = TesterAgent()
    outcome = await tester.run_command(["pytest", "-q"], cwd=Path("/repo"))
    assert outcome.ok is outcome.returncode == 0
    ```
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from furrow.agents.parsing import coerce_status, parse_json_response
from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM_PROMPT
from furrow.config import TaskModel, TestResult, get_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from furrow.config import Settings

__all__ = ["CommandResult", "TesterAgent"]

CANDIDATE_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("npm", "test", "--silent"),
    ("pnpm", "test", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)
"""Test runners probed in order, first one whose executable exists wins."""

DRAIN_TIMEOUT: float = 5.0
"""Seconds allowed for the output pipes to close after a process is killed."""

MAX_OUTPUT_CHARS: int = 8000
"""Captured test output embedded in the judge prompt; the tail is kept."""

MAX_RESULT_CHARS: int = 200
"""Per-task characters of a previous task result included in the judge prompt."""


class CommandResult(BaseModel):
    """Raw outcome of one test-runner invocation.

    Attributes:
        command: The argument vector that was executed.
        returncode: Exit status, or ``None`` when the process never reported
            one (killed on timeout, or never started).
        output: Everything captured from stdout and stderr, kept even for a
            timed-out run.
        timed_out: Whether the process was killed for exceeding
            ``Settings.test_timeout``.
        error: Why the process could not be run or completed, if applicable.
    """

    command: list[str]
    returncode: int | None
    output: str
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Return whether the run is an unambiguous pass.

        Returns:
            ``True`` only when the process started, was not killed for timeout
            and exited with status 0.
        """
        return not self.timed_out and self.error is None and self.returncode == 0


class TesterAgent:
    """Runs the test suite and asks the model to judge the outcome.

    Attributes:
        settings: Resolved configuration: the explicit ``settings`` argument
            when one is given, otherwise the client's, otherwise the
            process-global :func:`~furrow.config.get_settings` singleton.
        client: The LLM client used for the verdict request.
    """

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        """Resolve the configuration and the client to test with.

        Args:
            client: Client to reuse. Its ``settings`` are consulted only
                when no ``settings`` argument is supplied.
            settings: Configuration that takes precedence over the client's,
                and that builds the client when none is given.
        """
        self.settings = settings or (client.settings if client is not None else get_settings())
        self.client = client or LLMClient(settings=self.settings)

    async def run(self, goal: str, tasks: list[TaskModel] | None = None) -> TestResult:
        """Run the test suite and judge it against ``goal``.

        Args:
            goal: The objective the work was supposed to achieve.
            tasks: The tasks that were executed. Each contributes its id,
                description, status and result to the prompt, so the judge knows
                what was actually implemented.

        Returns:
            A :class:`~furrow.config.TestResult`. It is a failure when no runner
            could be detected, when the command could not run or timed out, or
            when the exit status contradicts the model's verdict.

        Raises:
            furrow.llm.LLMError: If the verdict request fails.
        """
        command = self._detect_command()
        if not command:
            reason = (
                f"No test runner found; set 'test_command' in Settings to run "
                f"the suite in {self.settings.workspace}"
            )
            return TestResult(passed=False, summary="No test runner found", failures=[reason])

        result = await self.run_command(command)
        prompt = self.build_prompt(goal, tasks, result)
        response = await self.client.complete(
            prompt,
            system=TESTER_SYSTEM_PROMPT,
            model=self.client.settings.tester_model,
        )
        try:
            verdict = parse_json_response(response, TestResult, "tester response")
        except ValueError as exc:
            return self._verdict_from_exit_status(result, detail=str(exc))
        return self._reconcile(verdict, result)

    async def run_command(self, command: list[str], cwd: Path | None = None) -> CommandResult:
        """Run ``command`` and capture its output.

        The process runs in ``cwd`` or in ``settings.workspace`` so relative
        paths such as ``./...`` mean the same thing as they would in a shell.
        On timeout the process is killed and whatever was captured up to that
        point is returned rather than discarded, because the tail of a hung run
        is exactly what explains the hang.

        Args:
            command: Argument vector to execute.
            cwd: Working directory; defaults to ``settings.workspace``.

        Returns:
            The exit status, captured output and failure reason.
        """
        if not command:
            return CommandResult(
                command=[],
                returncode=None,
                output="",
                error="No test command to run",
            )
        working_dir = cwd or self.settings.workspace
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return CommandResult(
                command=list(command),
                returncode=None,
                output="",
                error=f"could not run {shlex.join(command)}: {exc}",
            )

        stdout = asyncio.ensure_future(_read_stream(process.stdout))
        stderr = asyncio.ensure_future(_read_stream(process.stderr))
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=self.settings.test_timeout)
        except asyncio.TimeoutError:
            timed_out = True
            with suppress(ProcessLookupError):
                process.kill()
        output = await _collect(stdout, stderr)

        if timed_out:
            return CommandResult(
                command=list(command),
                returncode=process.returncode,
                output=output,
                timed_out=True,
                error=(
                    f"test command exceeded test_timeout of "
                    f"{self.settings.test_timeout:g}s and was killed"
                ),
            )
        return CommandResult(command=list(command), returncode=process.returncode, output=output)

    def detect_command(self) -> list[str]:
        """Return the test runner that :meth:`run` would use.

        Returns:
            The detected argument vector, empty when no runner is available.
        """
        return self._detect_command()

    def _detect_command(self) -> list[str]:
        """Choose the test runner to use.

        ``Settings.test_command`` wins when set. Otherwise the known runners are
        probed with :func:`shutil.which` so a missing toolchain is skipped
        without spawning processes that fail.

        Returns:
            The argument vector to run, or an empty list when no runner is
            available.
        """
        override = self.settings.test_command
        if override and override.strip():
            command = shlex.split(override)
            if command:
                return command
        for candidate in CANDIDATE_COMMANDS:
            if shutil.which(candidate[0]) is not None:
                return list(candidate)
        return []

    def build_prompt(
        self,
        goal: str,
        tasks: list[TaskModel] | None,
        result: CommandResult,
    ) -> str:
        """Assemble the judge prompt.

        Args:
            goal: The objective under test.
            tasks: Executed tasks, summarised so the judge knows what was built.
            result: The captured run.

        Returns:
            The prompt text: template, goal, task digest, exit status and output.
        """
        status = "timed out" if result.timed_out else f"exit status {result.returncode}"
        output = result.output.strip() or "(no output captured)"
        if len(output) > MAX_OUTPUT_CHARS:
            tail = output[-MAX_OUTPUT_CHARS:]
            output = f"(truncated to the last {MAX_OUTPUT_CHARS} characters)\n{tail}"
        lines = [
            TESTER_PROMPT,
            "",
            f"Goal: {goal}",
            "",
            "Work that was implemented:",
            self._task_digest(tasks or []),
            "",
            f"Command: {shlex.join(result.command) or '(none)'}",
            f"Status: {status}",
            "",
            "Test output:",
            output,
        ]
        if result.error:
            lines.extend(["", f"Runner error: {result.error}"])
        return "\n".join(lines)

    def _task_digest(self, tasks: list[TaskModel]) -> str:
        """Summarise executed tasks for the judge prompt.

        Args:
            tasks: Tasks produced by the planner and executed in this cycle.

        Returns:
            One block per task with its id, description, status and a truncated
            result, or a placeholder when no tasks were supplied.
        """
        if not tasks:
            return "(no task records were provided)"
        blocks: list[str] = []
        for task in tasks:
            status = coerce_status(task.status).value
            result = (task.result or "").strip().replace("\n", " ")
            if len(result) > MAX_RESULT_CHARS:
                result = f"{result[:MAX_RESULT_CHARS]}..."
            blocks.append(
                f"- id={task.id} status={status} "
                f"files=[{', '.join(task.files) or '-'}] "
                f"depends_on=[{', '.join(task.dependencies) or '-'}]\n"
                f"  {task.description}\n"
                f"  result: {result or '(no result recorded)'}"
            )
        return "\n".join(blocks)

    def _reconcile(self, verdict: TestResult, result: CommandResult) -> TestResult:
        """Refuse to report a pass the exit status contradicts.

        Args:
            verdict: The model's judgement.
            result: The captured run.

        Returns:
            ``verdict`` unchanged when it does not contradict the exit status,
            otherwise a failing result that names the real reason.
        """
        if not verdict.passed or result.ok:
            return verdict
        reason = result.error or f"test command exited with status {result.returncode}"
        failures = [*verdict.failures, reason]
        summary = verdict.summary.strip()
        note = f"verdict overruled by exit status: {reason}"
        return TestResult(
            passed=False,
            summary=f"{summary} ({note})" if summary else note,
            failures=failures,
        )

    def _verdict_from_exit_status(
        self,
        result: CommandResult,
        detail: str,
    ) -> TestResult:
        """Fall back to the exit status when the model's reply is unusable.

        Args:
            result: The captured run.
            detail: Why the model's reply could not be parsed.

        Returns:
            A :class:`~furrow.config.TestResult` derived purely from the exit
            status, with the parse failure recorded as a failure so a broken
            verdict is not mistaken for a green suite.
        """
        passed = result.ok
        reason = result.error or f"test command exited with status {result.returncode}"
        summary = (
            "Test command succeeded but the judge reply was not valid JSON."
            if passed
            else f"Test run failed ({reason}) and the judge reply was not valid JSON."
        )
        return TestResult(
            passed=passed,
            summary=summary,
            failures=[] if passed else [reason, detail],
        )


async def _read_stream(stream: asyncio.StreamReader | None) -> bytes:
    """Drain one subprocess pipe.

    Args:
        stream: The pipe to read, or ``None`` when it was not captured.

    Returns:
        The bytes read, empty when there was no stream.
    """
    if stream is None:
        return b""
    return await stream.read()


async def _collect(*readers: Awaitable[bytes]) -> str:
    """Await the pipe readers and render their contents as text.

    A descendant process may keep a pipe open after its parent exits, so the
    wait is bounded; whatever finished by then is used and the rest is dropped.

    Args:
        readers: The pending pipe reads, stdout first.

    Returns:
        The captured stdout followed by stderr.
    """
    tasks = [asyncio.ensure_future(reader) for reader in readers]
    done, pending = await asyncio.wait(tasks, timeout=DRAIN_TIMEOUT)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    parts: list[str] = []
    for task in tasks:
        if task in done and not task.cancelled():
            parts.append(task.result().decode("utf-8", errors="replace"))
    return "".join(parts)
