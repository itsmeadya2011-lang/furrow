"""Tester agent: run the project's real test suite and report the verdict.

The tester is an observer, never a participant. Two rules shape the code here:

* **Report only what happened.** The subprocess return code is carried through
  into the prompt and into the fallback verdict, and it outranks a reported
  pass: a model cannot turn a non-zero exit, a timeout, or an execution error
  into ``passed: true``. The override runs one way only — a green run the model
  calls a failure is kept. A missing test runner is a failing result rather than
  a fabricated pass.
* **A failing verdict must carry failures.** The orchestrator feeds
  ``"\\n".join(failures)`` into the next cycle's goal, so an empty failure list on
  a failing run silently destroys the goal and the next cycle starts inventing
  work. Every path out of :meth:`TesterAgent.run` therefore guarantees
  ``passed is False`` implies ``failures`` is non-empty.

Nothing in :meth:`TesterAgent.run` raises: one broken test run returns a failing
:class:`~furrow.config.TestResult` instead of killing the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
import re
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM
from furrow.config import TaskModel, TestResult
from furrow.config import settings as config_settings
from furrow.errors import StructuredOutputError
from furrow.llm import LLMClient
from furrow.parsing import parse_model

if TYPE_CHECKING:
    from furrow.config import Settings

_LOGGER = logging.getLogger(__name__)

#: Cap on the test output handed to the model, in characters.
MAX_TEST_OUTPUT_CHARS = 20_000

#: Cap on the tail quoted into a synthesised failure entry, in characters.
_TAIL_CHARS = 2_000

#: Cap on the number of diagnostic lines pulled out of a failing run.
_MAX_DIAGNOSTIC_LINES = 12

NO_RUNNER_SUMMARY = "No test runner found"

#: Files that mark a Python project, mapped to nothing -- the command is fixed
#: per marker family below.
_PYTHON_MARKERS = ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg")

_ERROR_HINT_RE = re.compile(
    r"(\berror\b|\bfail(ed|s|ure|ures)?\b|\btraceback\b|\bassert\w*\b|\bexception\b|\bpanic\b)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class CommandResult:
    """Outcome of one test-runner invocation."""

    command: list[str] = field(default_factory=list)
    exit_code: int | None = None
    output: str = ""
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True only when the command ran, finished in time, and exited zero."""
        return self.exit_code == 0 and not self.timed_out and self.error is None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, suitable for event payloads."""
        return {
            "command": list(self.command),
            "exit_code": self.exit_code,
            "output": self.output,
            "timed_out": self.timed_out,
            "error": self.error,
            "ok": self.ok,
        }


def _decode(raw: bytes | None) -> str:
    """Decode a stream, replacing undecodable bytes instead of raising."""
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def _diagnostic_lines(output: str, limit: int = _MAX_DIAGNOSTIC_LINES) -> list[str]:
    """Lines of a failing run most likely to identify the cause."""
    interesting: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= set("-=_*+~"):
            continue
        if _ERROR_HINT_RE.search(stripped) and stripped not in interesting:
            interesting.append(stripped)
        if len(interesting) >= limit:
            break
    return interesting


def _tail(output: str, limit: int = _TAIL_CHARS) -> str:
    """Last ``limit`` characters of ``output``, marked when elided."""
    text = output.strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return "... [earlier output omitted] ...\n" + text[-limit:]


class TesterAgent:
    """Runs the workspace test suite and turns the run into a ``TestResult``."""

    def __init__(
        self,
        client: LLMClient,
        *,
        settings: Settings | None = None,
        workspace: Path | str | None = None,
    ) -> None:
        self._client = client
        self._settings = settings if settings is not None else config_settings
        self._workspace = Path(workspace) if workspace else self._settings.workspace

    @property
    def workspace(self) -> Path:
        """The directory the test command is run from."""
        return self._workspace

    async def _discover_command(self) -> list[str] | None:
        """Pick the test command for this workspace, or ``None`` if there is none.

        ``settings.test_command`` wins when set. Otherwise a project marker file
        selects the command family and the executable is confirmed to exist, so
        an unsuitable guess never runs and its output is never reported.
        """
        configured = self._settings.test_command
        if configured and configured.strip():
            command = shlex.split(configured)
            if command:
                return command
            _LOGGER.warning("settings.test_command is whitespace only; falling back to discovery")

        if self._has_marker(_PYTHON_MARKERS):
            command = self._pytest_command()
            if command is not None:
                return command

        candidates = (
            ("package.json", ["npm", "test", "--silent"]),
            ("Cargo.toml", ["cargo", "test", "-q"]),
            ("go.mod", ["go", "test", "./..."]),
        )
        for marker, command in candidates:
            if self._has_marker((marker,)) and shutil.which(command[0]) is not None:
                return list(command)
        return None

    def _has_marker(self, markers: tuple[str, ...]) -> bool:
        """Whether any marker file exists at the workspace root."""
        return any((self._workspace / marker).is_file() for marker in markers)

    def _pytest_command(self) -> list[str] | None:
        """``pytest -q`` if pytest is runnable here, else ``None``."""
        if shutil.which("pytest") is not None:
            return ["pytest", "-q"]
        try:
            has_pytest = importlib.util.find_spec("pytest") is not None
        except (ImportError, ValueError):
            has_pytest = False
        if has_pytest:
            return [sys.executable, "-m", "pytest", "-q"]
        return None

    async def _run_command(
        self,
        command: list[str],
        *,
        timeout: int | None = None,
    ) -> CommandResult:
        """Run ``command`` in the workspace and report exactly what happened."""
        if not command:
            return CommandResult(command=[], error="no command to run")
        limit = timeout if timeout is not None else self._settings.test_timeout

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=self._workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            # FileNotFoundError (missing binary), NotADirectoryError, PermissionError...
            return CommandResult(command=list(command), error=f"{type(exc).__name__}: {exc}")

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=limit)
        except asyncio.TimeoutError:
            proc.kill()
            # Drain both pipes after killing, otherwise the child keeps running
            # as an orphan and its transport is never closed.
            stdout, stderr = await proc.communicate()
            output = _decode(stdout) + _decode(stderr)
            output += f"\n[command killed after exceeding the {limit}s timeout]"
            return CommandResult(
                command=list(command),
                exit_code=proc.returncode,
                output=output,
                timed_out=True,
                error=f"command timed out after {limit}s",
            )
        except asyncio.CancelledError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.communicate()
            raise

        return CommandResult(
            command=list(command),
            exit_code=proc.returncode,
            output=_decode(stdout) + _decode(stderr),
        )

    def _clip_output(self, output: str) -> str:
        """Cap the output handed to the model, saying so when it is elided."""
        if len(output) <= MAX_TEST_OUTPUT_CHARS:
            return output
        elided = len(output) - MAX_TEST_OUTPUT_CHARS
        return (
            output[:MAX_TEST_OUTPUT_CHARS]
            + f"\n... [truncated, {elided} more characters of test output]"
        )

    def _fallback_failures(self, result: CommandResult) -> list[str]:
        """Synthesise a non-empty failure list from a run the model did not describe."""
        command = " ".join(result.command) or "(no command)"
        if result.timed_out:
            header = f"test command `{command}` timed out after {self._settings.test_timeout}s"
        elif result.error is not None:
            header = f"test command `{command}` could not be executed: {result.error}"
        else:
            header = f"test command `{command}` exited with code {result.exit_code}"
        lines = _diagnostic_lines(result.output)
        failures = [header, *(f"{command}: {line}" for line in lines)]
        if not lines:
            tail = _tail(result.output)
            if tail:
                failures.append(f"{command} output: {tail}")
        return failures

    def _enforce_failures(self, verdict: TestResult, result: CommandResult) -> TestResult:
        """Guarantee ``passed is False`` implies ``failures`` carries real text.

        The orchestrator feeds ``"\\n".join(failures)`` into the next cycle's goal.
        A failing verdict with nothing usable in that list therefore reduces the
        goal to whitespace, and the next cycle invents work to do. When the model
        reports a failure but no usable failures, they are rebuilt from the exit
        code and the run's own output rather than passed through.

        A list of blank strings counts as empty: it is non-empty by ``len()`` but
        tells the next cycle nothing, which is the failure this guards against.
        """
        if verdict.passed or any(failure.strip() for failure in verdict.failures):
            return verdict
        failures = self._fallback_failures(result)
        _LOGGER.warning(
            "tester reported a failing run with no usable failures; synthesised %d entr%s",
            len(failures),
            "y" if len(failures) == 1 else "ies",
        )
        return verdict.model_copy(update={"failures": failures})

    def _veto_false_pass(self, verdict: TestResult, result: CommandResult) -> TestResult:
        """Never let a parseable ``passed: true`` outrank a failed command run.

        WHY the exit code wins here: "tests passed" is the loop's entire
        termination signal, so a model that wrongly reports success makes the
        orchestrator stop believing the goal is met. The subprocess is the ground
        truth we actually observed — if it exited non-zero, timed out, or could
        not be executed, the suite did not pass, whatever the model claims.

        The override is deliberately one-directional. The other direction is a
        legitimate judgement, not a bug: the command can exit 0 while the goal
        is still unmet (tests green, feature missing), and that failing verdict
        is preserved verbatim.

        Only ``passed`` is rewritten, so a model that reported real failures
        keeps them; when it left the list empty, :meth:`_enforce_failures` fills
        it from the exit code and the run's output.
        """
        if verdict.passed and not result.ok:
            command = " ".join(result.command) or "(no command)"
            if result.timed_out:
                because = f"timed out after {self._settings.test_timeout}s"
            elif result.error is not None:
                because = f"could not be executed: {result.error}"
            else:
                because = f"exited with code {result.exit_code}"
            _LOGGER.warning(
                "tester model reported a passing run but `%s` %s; forcing a failing verdict",
                command,
                because,
            )
            return verdict.model_copy(update={"passed": False})
        return verdict

    def _deterministic_verdict(self, result: CommandResult, note: str) -> TestResult:
        """Verdict straight from the exit code, used when the model's is unusable."""
        if result.ok:
            return TestResult(
                passed=True,
                summary=note or f"`{' '.join(result.command)}` exited with code 0.",
                failures=[],
            )
        failures = self._fallback_failures(result)
        return TestResult(
            passed=False,
            summary=note or f"`{' '.join(result.command)}` failed: {failures[0]}.",
            failures=failures,
        )

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the suite for ``goal`` and return the verdict. Never raises."""
        try:
            command = await self._discover_command()
            if command is None:
                return TestResult(
                    passed=False,
                    summary=NO_RUNNER_SUMMARY,
                    failures=[
                        f"no test runner found in {self._workspace}; "
                        "add FURROW_TEST_COMMAND to point at one"
                    ],
                )
            result = await self._run_command(command)
            verdict = await self._classify(goal, tasks, command, result)
        except asyncio.CancelledError:
            raise
        # Broad by design: a broken run must not kill the loop.
        except Exception as exc:
            _LOGGER.warning("test run failed: %s", exc)
            return TestResult(
                passed=False,
                summary=f"Test run could not be completed: {type(exc).__name__}: {exc}",
                failures=[f"test run could not be completed: {type(exc).__name__}: {exc}"],
            )
        return self._enforce_failures(verdict, result)

    async def _classify(
        self,
        goal: str,
        tasks: list[TaskModel],
        command: list[str],
        result: CommandResult,
    ) -> TestResult:
        """Ask the model to classify a completed run; fall back to the exit code."""
        if result.error is not None and not result.output:
            # The command never ran: there is nothing to classify.
            return self._deterministic_verdict(result, note=f"Test run failed: {result.error}.")

        prompt = TESTER_PROMPT.format(
            goal=goal,
            test_output=self._clip_output(result.output),
            exit_code=result.exit_code if result.exit_code is not None else "unknown",
            command=" ".join(command),
            files=self._describe_files(tasks),
        )
        try:
            response = await self._client.complete(
                prompt,
                system=TESTER_SYSTEM,
                model=self._settings.model_for("tester"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _LOGGER.warning("tester LLM call failed (%s); using the exit code", exc)
            return self._deterministic_verdict(
                result,
                note=f"The tester model could not be reached ({exc}); reporting the exit code.",
            )

        try:
            verdict = parse_model(response, TestResult)
        except StructuredOutputError as exc:
            # No substring matching on the raw text: a run that prints the word
            # "passed" inside a failure message would be reported as green.
            _LOGGER.warning("tester output unusable (%s); using the exit code", exc.message)
            return self._deterministic_verdict(
                result,
                note="The tester model did not return a usable verdict; reporting the exit code.",
            )
        # Parsing succeeded, so the model did report a verdict -- but a reported
        # pass still cannot overrule a run that failed on its own terms.
        return self._veto_false_pass(verdict, result)

    @staticmethod
    def _describe_files(tasks: list[TaskModel]) -> str:
        """The files the planner expected to change, for the prompt context."""
        files = sorted({path for task in tasks for path in task.files})
        return ", ".join(files) if files else "(none declared by the planner)"
