"""Tester agent: runs the project's test suite and asks the LLM to diagnose it.

The pass/fail verdict belongs to the test runner, not to the model. Reading
``proc.returncode`` is the only trustworthy signal available: a substring test on the
model's prose ("passed" in ``response.lower()``) matches "1 passed, 3 failed" and
"no tests passed" alike, which silently marks a red suite green. The model's job is
diagnosis -- which tests failed and why -- so that the next cycle knows what to fix.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pydantic import ValidationError

from furrow.agents.jsonutil import JSONExtractionError, extract_json_object, truncate_output
from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient, LLMError

if TYPE_CHECKING:
    from pathlib import Path

    from furrow.config import Settings

# Conservative characters-per-token ratio for test logs, used to keep the prompt
# inside the per-request budget of ``llm_max_tokens``.
_CHARS_PER_TOKEN = 2
_MAX_OUTPUT_CHARS = 16_000
_MAX_RESULT_CHARS = 400
_MAX_TASKS_IN_PROMPT = 20
# How long to wait for a killed runner's pipes to close while draining its output.
_DRAIN_TIMEOUT = 2.0

# Lines that look like a test failure. Used only to seed ``failures`` when the model
# reply is unusable prose; it is never used to decide pass/fail.
_FAILURE_MARKERS = (
    "FAILED",
    "FAIL:",
    "FAIL ",
    "ERROR:",
    "ERROR ",
    "E   ",
    "AssertionError",
    "Traceback",
    "assertion failed",
    "error TS",
    "✕",
    "✗",
    "not ok",
)
_MAX_EXTRACTED_FAILURES = 10


@dataclass(frozen=True)
class TestRun:
    """The raw outcome of probing one test-runner command.

    Attributes:
        command: the command that was attempted (empty when nothing was found).
        returncode: the runner's exit status, or ``None`` if it never ran.
        output: merged stdout/stderr, with a timeout note appended when applicable.
        timed_out: whether the runner was killed after ``settings.test_timeout``.
        started: whether a runner process actually launched. ``False`` means the
            binary is not installed (keep probing) or could not be started at all.
        error: a real, non-"missing binary" failure encountered while probing.
    """

    command: tuple[str, ...] = ()
    returncode: int | None = None
    output: str = ""
    timed_out: bool = False
    started: bool = False
    error: str | None = None


class TesterAgent:
    """Runs the test suite in the workspace and reports what the runner decided."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        # ``LLMClient`` resolves ``settings=None`` to ``get_settings()``, so passing
        # None through explicitly is safe. (The old client stored None verbatim, which
        # made every later ``self.client.settings.tester_model`` an AttributeError.)
        self.client = client or LLMClient(settings=settings)

    @property
    def settings(self) -> Settings:
        """The settings used for testing, read from the client."""
        return self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the suite and return the runner's verdict plus the model's diagnosis."""
        probe = await self._run_tests()

        if not probe.started:
            if probe.error:
                # A real OS-level failure, not a missing runner: surface it instead of
                # pretending there is nothing to verify.
                summary = f"Could not run the test suite: {probe.error}"
                return TestResult(passed=False, summary=summary, failures=[probe.error])
            # Deliberate: "nothing to verify" is treated as a pass. Reporting it as a
            # failure makes the orchestrator rewrite the goal to fix tests that do not
            # exist and spin forever on projects without a suite.
            return TestResult(
                passed=True,
                summary="No test runner detected; nothing to verify",
                failures=[],
            )

        # The runner owns the verdict. A timeout counts as failure: the suite did not
        # finish, so nothing was verified.
        passed = probe.returncode == 0 and not probe.timed_out
        output = truncate_output(probe.output, self._output_budget())
        prompt = self._build_prompt(goal, tasks, probe, output)
        try:
            response = await self.client.complete(
                prompt,
                model=self.settings.tester_model,
                system=TESTER_SYSTEM,
            )
        except LLMError as exc:
            # A judge outage must not mask the runner's verdict.
            return self._result(
                passed=passed,
                summary=(
                    f"Tests {'passed' if passed else 'failed'} (exit code "
                    f"{probe.returncode}); diagnosis unavailable: {exc}"
                ),
                failures=self._failure_markers(output) if not passed else [],
            )

        return self._diagnose(response, probe, passed, output)

    # -- test runner -------------------------------------------------------

    async def _run_tests(self) -> TestRun:
        """Probe the known test runners and return the first one that actually ran."""
        workspace = self.settings.workspace
        if not workspace.is_dir():
            return TestRun(
                error=f"workspace '{workspace}' is not an existing directory"
            )

        errors: list[str] = []
        for command in self._candidates():
            run = await self._execute(command)
            if run.error:
                # A real failure (permissions, fork errors, ...): remember it so the
                # result does not degrade into "no runner detected".
                errors.append(f"{' '.join(command)}: {run.error}")
                continue
            if not run.started:
                continue  # runner binary not installed; try the next candidate
            if errors:
                # Keep the earlier probe failures visible in the transcript.
                note = f"[furrow] earlier probe failures: {'; '.join(errors)}"
                run = replace(run, output=f"{note}\n{run.output}")
            return run

        return TestRun(error="; ".join(errors) if errors else None)

    @staticmethod
    def _candidates() -> list[tuple[str, ...]]:
        """Test-runner commands to probe, in order of preference."""
        return [
            ("pytest", "-q"),
            ("python3", "-m", "pytest", "-q"),
            ("python", "-m", "pytest", "-q"),
            ("npm", "test", "--", "--silent"),
            ("pnpm", "test", "--", "--silent"),
            ("yarn", "test", "--silent"),
            ("cargo", "test", "-q"),
            ("go", "test", "./..."),
        ]

    async def _execute(self, command: Sequence[str]) -> TestRun:
        """Run one candidate command in the workspace, capturing merged output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._cwd(),
            )
        except FileNotFoundError:
            # The runner is simply not installed: keep probing.
            return TestRun(command=tuple(command))
        except OSError as exc:
            return TestRun(command=tuple(command), error=f"{type(exc).__name__}: {exc}")

        reader = asyncio.ensure_future(proc.stdout.read() if proc.stdout else _empty())
        try:
            raw = await asyncio.wait_for(
                asyncio.shield(reader), timeout=self._timeout()
            )
        except asyncio.TimeoutError:
            # kill() then wait() so the child is reaped instead of lingering as a
            # zombie; the shield kept `reader` alive so its buffered output survives.
            proc.kill()
            raw = await self._drain(reader)
            await self._reap(proc)
            note = (
                f"\n[furrow] test runner timed out after {self._timeout()}s "
                f"and was killed: {' '.join(command)}"
            )
            return TestRun(
                command=tuple(command),
                returncode=proc.returncode,
                output=_decode(raw) + note,
                timed_out=True,
                started=True,
            )
        except (OSError, ValueError) as exc:  # pragma: no cover - pipe teardown races
            proc.kill()
            await self._reap(proc)
            return TestRun(command=tuple(command), error=f"{type(exc).__name__}: {exc}")

        await self._reap(proc)
        return TestRun(
            command=tuple(command),
            returncode=proc.returncode,
            output=_decode(raw),
            started=True,
        )

    @staticmethod
    async def _reap(proc: asyncio.subprocess.Process) -> None:
        """Wait for the child and close its transport.

        ``Process.kill()`` bypasses the pipe teardown asyncio performs on a normal
        exit, so without the explicit close the transport is left unclosed: its file
        descriptors linger and ``__del__`` complains once the event loop shuts down.
        """
        await proc.wait()
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            with contextlib.suppress(RuntimeError, OSError):
                transport.close()

    @staticmethod
    async def _drain(reader: asyncio.Future[bytes]) -> bytes:
        """Best-effort read of whatever a killed runner had already produced."""
        try:
            return await asyncio.wait_for(asyncio.shield(reader), timeout=_DRAIN_TIMEOUT)
        except (asyncio.TimeoutError, asyncio.CancelledError, OSError):
            reader.cancel()
            return b""

    # -- prompting and diagnosis -------------------------------------------

    def _build_prompt(
        self,
        goal: str,
        tasks: list[TaskModel],
        probe: TestRun,
        output: str,
    ) -> str:
        """Assemble the judge prompt: goal, task results, and the runner output."""
        sections = [
            TESTER_PROMPT,
            f"\nGoal: {goal}",
            self._describe_run(probe, output),
            self._describe_tasks(tasks),
        ]
        return "\n".join(s for s in sections if s)

    @staticmethod
    def _describe_run(probe: TestRun, output: str) -> str:
        """State the runner's verdict explicitly; the model must not re-derive it."""
        if probe.timed_out:
            verdict = "TIMED OUT (killed) - treat as a failure"
        elif probe.returncode == 0:
            verdict = "exit code 0 - PASSED"
        else:
            verdict = f"exit code {probe.returncode} - FAILED"
        return (
            f"\nRunner command: {' '.join(probe.command)}\n"
            f"Runner verdict: {verdict}\n"
            f"\nTest output:\n{output or '(no output)'}"
        )

    @staticmethod
    def _describe_tasks(tasks: list[TaskModel]) -> str:
        """Summarise the tasks that were implemented, so failures can be attributed."""
        if not tasks:
            return "\nTasks implemented: (none reported)"
        lines = ["\nTasks implemented in this cycle:"]
        for task in tasks[:_MAX_TASKS_IN_PROMPT]:
            status = task.status.value if hasattr(task.status, "value") else str(task.status)
            # A caller may store either a plain summary or a WorkerResult; str() on the
            # latter yields its summary, so both shapes render identically.
            raw = task.result or ""
            result = truncate_output(raw if isinstance(raw, str) else str(raw), _MAX_RESULT_CHARS)
            lines.append(f"- [{task.id}] ({status}) {task.description}")
            if result:
                lines.append(f"  result: {result}")
        if len(tasks) > _MAX_TASKS_IN_PROMPT:
            lines.append(f"- ... and {len(tasks) - _MAX_TASKS_IN_PROMPT} more tasks")
        return "\n".join(lines)

    def _diagnose(
        self,
        response: str,
        probe: TestRun,
        passed: bool,
        output: str,
    ) -> TestResult:
        """Turn the model reply into a :class:`TestResult`, keeping the runner verdict."""
        try:
            data = extract_json_object(response)
            result = TestResult.model_validate(data)
        except (JSONExtractionError, ValidationError):
            # Unusable reply: fall back to the runner verdict plus a failure-marker
            # scan of the output. The markers seed the failure list only -- pass/fail
            # is never inferred from text.
            failures = [] if passed else self._failure_markers(output)
            return self._result(
                passed=passed,
                summary=truncate_output(
                    response.strip() or self._default_summary(probe), 2_000
                ),
                failures=failures,
            )
        # The runner's exit code is authoritative, whatever the model claimed.
        result.passed = passed
        return result

    @staticmethod
    def _result(passed: bool, summary: str, failures: list[str]) -> TestResult:
        return TestResult(passed=passed, summary=summary, failures=failures)

    @staticmethod
    def _default_summary(probe: TestRun) -> str:
        if probe.timed_out:
            return f"Test runner timed out: {' '.join(probe.command)}"
        return (
            f"Test runner {' '.join(probe.command)} exited with code {probe.returncode}"
        )

    @staticmethod
    def _failure_markers(output: str) -> list[str]:
        """Extract likely-failing lines from runner output (failure seeding only)."""
        found: list[str] = []
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(marker in stripped for marker in _FAILURE_MARKERS) and stripped not in found:
                found.append(stripped[:_MAX_RESULT_CHARS])
            if len(found) >= _MAX_EXTRACTED_FAILURES:
                break
        return found

    def _timeout(self) -> float:
        """Seconds to allow the runner before it is killed."""
        return float(max(1, self.settings.test_timeout))

    def _output_budget(self) -> int:
        """Character budget for runner output embedded in the judge prompt."""
        return max(1_000, min(_MAX_OUTPUT_CHARS, self.settings.llm_max_tokens * _CHARS_PER_TOKEN))

    def _cwd(self) -> Path:
        """Working directory for the runner: the configured workspace."""
        return self.settings.workspace


async def _empty() -> bytes:
    return b""


def _decode(raw: bytes | str | None) -> str:
    """Decode runner output, replacing undecodable bytes instead of raising."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="replace")
