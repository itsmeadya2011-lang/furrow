from __future__ import annotations

import asyncio
import shlex
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from furrow.agents.planner import extract_json
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Wording the orchestrator string-matches to treat a run as "skipped", not failed.
NO_RUNNER_OUTPUT = "No test runner found."
NO_RUNNER_SUMMARY = "No test runner found; skipping verification."

# Exit codes borrowed from coreutils/timeout conventions.
TIMEOUT_RETURNCODE = 124
NOT_FOUND_RETURNCODE = 127

MAX_FAILURES = 20
FAILURE_MARKERS = ("FAILED", "Error", "error:", "assert", "Traceback")
EMPTY_OUTPUT_SUMMARY = "Test command produced no output."

# Upper bound for the ``-c "import <module>"`` probe. The probe only has to prove
# that an import works, so a slow interpreter means "not usable", not "wait
# longer": capped well below ``settings.test_timeout`` so a wrong probe can never
# turn into a hang of its own.
PROBE_TIMEOUT = 30

# Candidate runners in priority order. A candidate is skipped only when it cannot
# run at all -- either its executable is genuinely absent from PATH, or it is a
# ``<interpreter> -m <tool>`` runner whose tool is not importable by that
# interpreter. Once one runs, its result stands even when it fails, so a Go repo
# with pytest installed cannot hijack the gate.
RUNNER_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("python3", "-m", "pytest", "-q"),
    ("npm", "test", "--", "--silent"),
    ("pnpm", "test", "--", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)


def _module_of(candidate: Sequence[str]) -> str | None:
    """The ``<tool>`` of a ``<interpreter> -m <tool> ...`` candidate, else ``None``."""
    if len(candidate) >= 3 and candidate[1] == "-m" and not candidate[2].startswith("-"):
        return candidate[2]
    return None


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one test-command invocation."""

    output: str
    returncode: int
    command: list[str] | None = None
    found: bool = True


def _decode(raw: bytes) -> str:
    return raw.decode(errors="replace")


def extract_failure_lines(output: str) -> list[str]:
    """Failure-ish lines from raw test output, capped at :data:`MAX_FAILURES`."""
    lines = [
        line.strip()
        for line in output.splitlines()
        if any(marker in line for marker in FAILURE_MARKERS)
    ]
    return [line for line in lines if line][:MAX_FAILURES]


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        result = await self._run_tests()
        if not result.found:
            # Nothing could be executed. Do not spend an LLM call on it, and do
            # not pretend the suite is green either.
            return TestResult(passed=True, summary=NO_RUNNER_SUMMARY, failures=[], skipped=True)

        # The subprocess exit code is the only source of truth about pass/fail.
        exit_ok = result.returncode == 0
        prompt = self._build_prompt(goal, tasks, result, exit_ok)
        try:
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        except Exception:
            # A summarisation failure must never hide the exit code.
            return self._from_exit_code(result, exit_ok)

        try:
            data = extract_json(response)
        except ValueError:
            return self._from_exit_code(result, exit_ok)

        summary = str(data.get("summary") or "").strip() or result.output.strip()
        return TestResult(
            # Never let the model flip the verdict: only the exit code decides.
            passed=exit_ok,
            summary=summary or EMPTY_OUTPUT_SUMMARY,
            failures=self._normalise_failures(data.get("failures"))
            or extract_failure_lines(result.output),
        )

    def _build_prompt(
        self,
        goal: str,
        tasks: list[TaskModel],
        result: CommandResult,
        exit_ok: bool,
    ) -> str:
        command = " ".join(result.command or [])
        verdict = "zero, so the suite passed" if exit_ok else "NON-ZERO, so the suite FAILED"
        sections = [
            TESTER_PROMPT,
            "",
            f"Goal: {goal}",
            "",
            f"Command: {command}",
            f"Exit code: {result.returncode} ({verdict})",
            "",
            "Test output:",
            result.output or EMPTY_OUTPUT_SUMMARY,
        ]
        if tasks:
            listed = "\n".join(
                f"- [{task.id}] {task.description} (status: {task.status})" for task in tasks
            )
            sections.extend(["", "Tasks that were just implemented:", listed])
        return "\n".join(sections)

    @staticmethod
    def _normalise_failures(raw: object) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            items: list[object] = [raw]
        elif isinstance(raw, (list, tuple, set, frozenset)):
            items = list(raw)
        else:
            items = [raw]
        return [text for text in (str(item).strip() for item in items) if text][:MAX_FAILURES]

    def _from_exit_code(self, result: CommandResult, exit_ok: bool) -> TestResult:
        return TestResult(
            passed=exit_ok,
            summary=result.output.strip() or EMPTY_OUTPUT_SUMMARY,
            failures=extract_failure_lines(result.output),
        )

    async def _run_tests(self) -> CommandResult:
        settings = self.client.settings
        cwd = str(settings.workspace)
        if settings.test_command:
            # An explicit command is authoritative: never fall through to guessing.
            return await self._run_command(shlex.split(settings.test_command), cwd)
        for candidate in RUNNER_CANDIDATES:
            if shutil.which(candidate[0]) is None:
                continue
            module = _module_of(candidate)
            # The executable exists, but ``<interpreter> -m <tool>`` also needs that
            # interpreter to be able to import the tool. In this container
            # ``/usr/bin/python3`` has no pytest, so running it anyway yields a red
            # gate for a suite that never existed. Probe before committing.
            if module is not None and not await self._probe_module(candidate[0], module, cwd):
                continue
            # The runner works: its exit code is the verdict, even when non-zero.
            return await self._run_command(list(candidate), cwd)
        return CommandResult(output=NO_RUNNER_OUTPUT, returncode=0, command=None, found=False)

    async def _probe_module(self, interpreter: str, module: str, cwd: str) -> bool:
        """Whether ``interpreter`` can ``import module`` within the probe budget."""
        settings = self.client.settings
        result = await self._run_command(
            [interpreter, "-c", f"import {module}"],
            cwd,
            timeout=min(settings.test_timeout, PROBE_TIMEOUT),
        )
        return result.returncode == 0

    async def _run_command(
        self, cmd: list[str], cwd: str, timeout: int | None = None
    ) -> CommandResult:
        printable = " ".join(cmd)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return CommandResult(
                output=f"Failed to run {printable!r}: {exc}",
                returncode=NOT_FOUND_RETURNCODE,
                command=cmd,
            )

        if timeout is None:
            timeout = self.client.settings.test_timeout
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            return await self._on_timeout(proc, cmd, printable, timeout)
        return CommandResult(
            output=_decode(stdout) + _decode(stderr),
            returncode=int(proc.returncode or 0),
            command=cmd,
        )

    @staticmethod
    async def _on_timeout(
        proc: asyncio.subprocess.Process,
        cmd: list[str],
        printable: str,
        timeout: int,
    ) -> CommandResult:
        # Kill *and* reap: a killed-but-unwaited child is a zombie that also
        # throws away whatever it managed to print before the deadline.
        proc.kill()
        partial = ""
        try:
            stdout, stderr = await proc.communicate()
        except OSError:
            pass
        else:
            partial = _decode(stdout) + _decode(stderr)
        note = f"[{printable} timed out after {timeout}s and was killed]"
        return CommandResult(
            output=f"{partial}\n{note}".strip(),
            returncode=TIMEOUT_RETURNCODE,
            command=cmd,
        )
