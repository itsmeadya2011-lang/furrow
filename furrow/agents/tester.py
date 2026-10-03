from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    from furrow.config import Settings

logger = logging.getLogger(__name__)

# A hung suite must not stall the loop forever; the run is reported as an
# explicit timeout failure instead of being silently skipped.
TEST_COMMAND_TIMEOUT = 120
LINT_COMMAND_TIMEOUT = 120
# Killing and draining a timed-out run needs its own bound.
KILL_GRACE_SECONDS = 10
# Test output goes straight back into a prompt, so it is capped. The tail is
# kept because that is where the actual failure lives.
MAX_OUTPUT_CHARS = 8000
MAX_FAILURE_LINES = 40
MAX_LLM_FAILURES = 20

# Ordered candidates; the first one whose executable exists is the run.
TEST_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("npm", "test", "--", "--silent"),
    ("pnpm", "test", "--", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)

# Verification is more than "tests pass": ruff and mypy are dev dependencies.
LINT_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("ruff", "check", "."),
    ("mypy", "."),
)

NO_RUNNER_SUMMARY = (
    "No test runner could be executed, so nothing was verified. "
    "This run is UNVERIFIED."
)


@dataclass
class CommandResult:
    """Outcome of one subprocess run. The return code is the authority."""

    command: tuple[str, ...]
    timeout: int = TEST_COMMAND_TIMEOUT
    available: bool = True
    returncode: int | None = None
    output: str = ""
    timed_out: bool = False
    error: str | None = None
    truncated: bool = False

    @property
    def label(self) -> str:
        return " ".join(self.command)

    @property
    def passed(self) -> bool:
        return (
            self.available
            and not self.timed_out
            and self.error is None
            and self.returncode == 0
        )

    def summary_line(self) -> str:
        if not self.available:
            return f"`{self.label}` not available ({self.error or 'executable not found'})"
        if self.timed_out:
            return f"`{self.label}` timed out after {self.timeout}s"
        if self.error is not None:
            return f"`{self.label}` failed to run ({self.error})"
        suffix = " [output truncated]" if self.truncated else ""
        return f"`{self.label}` exit {self.returncode}{suffix}"

    def failure_detail(self) -> str:
        """A single self-contained failure entry for the orchestrator."""
        if self.timed_out:
            return f"{self.summary_line()} - the suite never finished."
        if self.error is not None:
            return self.summary_line()
        return f"{self.summary_line()}\n{_tail(self.output, MAX_FAILURE_LINES)}"


class TesterAgent:
    """Runs the project test command (plus lint/typecheck) and reports the result."""

    # Not a pytest test class: pytest collects anything named ``Test*`` and
    # warns about the constructor. The public name is part of the API (the
    # orchestrator, agents package and web server all import it), so opt out
    # explicitly instead of renaming.
    __test__ = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    # --------------------------------------------------------------------- run

    async def run(
        self,
        goal: str,
        tasks: list[TaskModel],
        *,
        skip_lint: bool = False,
        test_command: Sequence[str] | None = None,
    ) -> TestResult:
        commands = self._test_commands(test_command)
        explicit = test_command is not None or self._configured_test_command() is not None

        test_result = await self._first_runnable(commands, TEST_COMMAND_TIMEOUT)
        if test_result is None or not test_result.available:
            # Nothing was executed. Reporting success here would let the loop
            # declare a goal verified that it never ran.
            return self._no_runner_result(commands, explicit)

        failures: list[str] = []
        if not test_result.passed:
            failures.append(test_result.failure_detail())

        lint_results: list[CommandResult] = []
        if not skip_lint:
            for command in LINT_COMMANDS:
                lint_result = await self._execute(command, LINT_COMMAND_TIMEOUT)
                if not lint_result.available:
                    logger.debug("Skipping %s: %s", lint_result.label, lint_result.error)
                    continue
                lint_results.append(lint_result)
                if not lint_result.passed:
                    failures.append(f"Lint/typecheck failed - {lint_result.failure_detail()}")

        passed = test_result.passed and all(result.passed for result in lint_results)
        lines = [test_result.summary_line(), *(r.summary_line() for r in lint_results)]
        summary = f"{' | '.join(lines)} ({len(tasks)} task(s) this cycle)"

        return await self._summarise(
            goal=goal,
            passed=passed,
            summary=summary,
            failures=failures,
            output=test_result.output,
            results=[test_result, *lint_results],
        )

    # --------------------------------------------------------------- execution

    async def _first_runnable(
        self,
        commands: Sequence[tuple[str, ...]],
        timeout: int,
    ) -> CommandResult | None:
        """Run the first candidate whose executable exists.

        A candidate that actually runs IS the result, even when it fails; we
        never fall through to the next runner on a non-zero exit code.
        """
        for command in commands:
            result = await self._execute(tuple(command), timeout)
            if result.available:
                return result
            logger.debug("Test candidate %s unavailable: %s", result.label, result.error)
        return None

    async def _execute(self, command: tuple[str, ...], timeout: int) -> CommandResult:
        result = CommandResult(command=command, timeout=timeout)
        workspace = getattr(self.client.settings, "workspace", ".")
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            # No such executable: this candidate simply is not installed.
            result.available = False
            result.error = f"{command[0]} is not installed or not on PATH"
            return result
        except OSError as exc:
            # Anything else (bad cwd, permissions, ...) is a genuine failure
            # and must not be mistaken for "no runner installed".
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return await self._kill(proc, result)

        text = stdout.decode(errors="replace") + stderr.decode(errors="replace")
        result.output, result.truncated = _truncate(text)
        result.returncode = proc.returncode
        return result

    async def _kill(
        self,
        proc: asyncio.subprocess.Process,
        result: CommandResult,
    ) -> CommandResult:
        """Kill, drain and reap a timed-out run; the timeout is the failure."""
        result.timed_out = True
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            stdout, stderr = await asyncio.wait_for(
                asyncio.gather(_read(proc.stdout), _read(proc.stderr)),
                timeout=KILL_GRACE_SECONDS,
            )
            result.output, result.truncated = _truncate(
                stdout.decode(errors="replace") + stderr.decode(errors="replace")
            )
        # Reap the child, otherwise a killed process lingers as a zombie.
        with contextlib.suppress(Exception):
            await proc.wait()
        result.returncode = proc.returncode
        return result

    # ---------------------------------------------------------------- verdicts

    def _no_runner_result(
        self,
        commands: Sequence[tuple[str, ...]],
        explicit: bool,
    ) -> TestResult:
        tried = ", ".join(" ".join(command) for command in commands) or "<none>"
        summary = (
            f"Configured test command could not be executed ({tried}). "
            "Nothing was verified; this run is UNVERIFIED."
            if explicit
            else f"{NO_RUNNER_SUMMARY} Tried: {tried}."
        )
        return TestResult(
            passed=False,
            summary=summary,
            failures=[f"No test runner found (tried: {tried}); the cycle is unverified."],
        )

    async def _summarise(
        self,
        *,
        goal: str,
        passed: bool,
        summary: str,
        failures: list[str],
        output: str,
        results: list[CommandResult],
    ) -> TestResult:
        """Let the model summarise only; ``passed`` always comes from exit codes."""
        verdict = "PASSED" if passed else "FAILED"
        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Commands run: {'; '.join(result.summary_line() for result in results)}\n"
            f"Authoritative verdict (from exit codes): {verdict}\n"
            f"Run summary: {summary}\n\n"
            f"Output (tail, may be truncated):\n{output}\n"
            "Never contradict the verdict above: summarise it and list individual failures."
        )

        try:
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        except Exception as exc:  # noqa: BLE001 - the LLM must never decide the verdict
            logger.warning("Tester summarisation failed; using the raw run result: %s", exc)
            return TestResult(passed=passed, summary=summary, failures=failures)

        data = _parse_summary(response)
        if data is None:
            logger.debug("Tester reply contained no usable JSON summary")
            return TestResult(passed=passed, summary=summary, failures=failures)

        llm_summary = data.get("summary")
        if isinstance(llm_summary, str) and llm_summary.strip():
            summary = f"{summary} — {llm_summary.strip()}"

        if passed:
            if data.get("passed") is False:
                logger.debug("Tester reported failure on a zero exit code; exit code wins")
            return TestResult(passed=True, summary=summary, failures=failures)

        for message in _string_list(data.get("failures"))[:MAX_LLM_FAILURES]:
            if message not in failures:
                failures.append(f"Reported by tester: {message}")
        return TestResult(passed=False, summary=summary, failures=failures)

    # --------------------------------------------------------------- settings

    def _configured_test_command(self) -> tuple[str, ...] | None:
        """An explicit command from settings, if such a field ever exists."""
        value = getattr(self.client.settings, "test_command", None)
        if isinstance(value, str) and value.strip():
            return tuple(value.split())
        if isinstance(value, (list, tuple)) and value:
            return tuple(str(item) for item in value)
        return None

    def _test_commands(
        self,
        test_command: Sequence[str] | None,
    ) -> tuple[tuple[str, ...], ...]:
        if test_command is not None:
            return (tuple(str(part) for part in test_command),)
        configured = self._configured_test_command()
        if configured is not None:
            return (configured,)
        return TEST_COMMANDS


# -------------------------------------------------------------------- helpers


def _parse_summary(response: str) -> dict[str, Any] | None:
    """Best-effort JSON object from a model reply (``extract_json`` first)."""
    text = (response or "").strip()
    if not text:
        return None
    try:
        data = json.loads(extract_json(text))
    except ValueError:
        # Fallback heuristic: first ``{`` up to the last ``}``.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
        elif item is not None and not isinstance(item, (list, dict, tuple, set)):
            items.append(str(item))
    return items


def _truncate(text: str, limit: int | None = None) -> tuple[str, bool]:
    """Keep the tail of ``text`` (where failures live), capped at ``limit``."""
    cap = MAX_OUTPUT_CHARS if limit is None else limit
    if len(text) <= cap:
        return text, False
    dropped = len(text) - cap
    return f"...[{dropped} earlier characters truncated]...\n{text[-cap:]}", True


def _tail(text: str, lines: int) -> str:
    if not text.strip():
        return ""
    parts = text.strip().splitlines()
    if len(parts) <= lines:
        return "\n".join(parts)
    return "\n".join([f"...[{len(parts) - lines} earlier lines omitted]...", *parts[-lines:]])


async def _read(stream: asyncio.StreamReader | None) -> bytes:
    if stream is None:
        return b""
    return await stream.read()