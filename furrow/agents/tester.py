from __future__ import annotations

import asyncio
import logging
import shlex
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from furrow.agents.parsing import (
    coerce_test_result,
    excerpt,
    looks_like_failure,
    looks_like_pass,
    parse_json_response,
)
from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from collections.abc import Sequence

    from furrow.config import Settings

logger = logging.getLogger(__name__)

NO_TEST_RUNNER = "No test runner found."
NO_USABLE_TEST_RUNNER = "No usable test runner."

# Ordered by how likely each ecosystem is to be the intended one. The first
# candidate that starts, produces a verdict, and actually collected tests wins.
_TEST_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("python3", "-m", "pytest", "-q"),
    ("npm", "test", "--", "--silent"),
    ("pnpm", "test", "--", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)

# Run after the tests; skipped silently when the tool is not installed so a
# project without ruff/mypy is not failed for not having them.
_STATIC_CHECKS: tuple[tuple[str, ...], ...] = (
    ("ruff", "check", "."),
    ("mypy", "furrow"),
)

# Output that means the runner started but verified nothing: not a pass and not
# a failure, so fall through to the next candidate instead of reporting a bogus
# result. (marker, reason)
_INCONCLUSIVE_MARKERS: tuple[tuple[str, str], ...] = (
    ("no tests ran", "collected no tests"),
    ("no tests collected", "collected no tests"),
    ("collected 0 items", "collected no tests"),
    ("0 tests", "collected no tests"),
    ("no test files", "no test files"),
    ("no tests to run", "collected no tests"),
    ("could not read package.json", "not a node project"),
    ("couldn't find a package.json", "not a node project"),
    ("find a package.json file", "not a node project"),
    ("no package.json", "not a node project"),
    ("no_importer_manifest_found", "not a node project"),
    ("could not find `cargo.toml`", "not a rust project"),
    ("could not find cargo.toml", "not a rust project"),
    ("go.mod file not found", "not a go project"),
    ("no go.mod file found", "not a go project"),
)

MAX_OUTPUT_CHARS = 8000
_MAX_TASK_LINES = 12
_MAX_FAILURE_LINES = 10

# Both markers mean nothing was actually verified, so the cycle reports that
# instead of calling it a test failure.
_NOTHING_VERIFIED_PREFIXES = (NO_TEST_RUNNER, NO_USABLE_TEST_RUNNER)


class TesterAgent:
    """Runs the project's checks and turns their output into a :class:`TestResult`.

    ``__test__ = False`` tells pytest not to collect this class purely because
    its name starts with ``Test`` -- the same marker :class:`TestResult` uses,
    applied here rather than in a global pytest filter.
    """

    __test__ = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if settings is not None:
            resolved = settings
        elif client is not None:
            resolved = client.settings
        else:
            resolved = default_settings
        self.client = client or LLMClient(settings=resolved)
        self.settings: Settings = resolved

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the checks for this cycle and ask the model for a verdict."""
        try:
            test_output = await self._run_tests()
        except (OSError, ValueError) as exc:
            logger.warning("could not run project checks: %s", exc)
            return TestResult(
                passed=False,
                summary=f"Could not run project checks: {exc}",
                failures=[str(exc)],
            )

        if test_output.startswith(_NOTHING_VERIFIED_PREFIXES):
            headline = _headline([test_output])
            logger.info("nothing was verified under %s: %s", self.settings.workspace, headline)
            return TestResult(
                passed=True,
                summary=f"{headline} Nothing was verified this cycle.",
                failures=[],
            )

        prompt = self._build_prompt(goal, tasks, test_output)
        response = await self.client.complete(
            prompt,
            system=TESTER_SYSTEM_PROMPT,
            model=self.settings.tester_model,
        )
        return self._verdict(response)

    async def _run_tests(self) -> str:
        """Return the captured output of the project's checks.

        ``settings.test_command`` wins and is run verbatim. Otherwise candidates
        are probed in order: a missing binary or a run that collected nothing is
        skipped, and the first real result is returned together with the lint and
        type-check output. A timeout is reported, not swallowed.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(1, self.settings.test_timeout)

        explicit = (self.settings.test_command or "").strip()
        if explicit:
            return await self._run_explicit_command(explicit, deadline)
        return await self._autodetect(deadline)

    async def _run_explicit_command(self, command: str, deadline: float) -> str:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            logger.warning("settings.test_command %r is not a valid command: %s", command, exc)
            return f"$ {command}\nFAILED TO START: {exc}"
        if not argv:
            logger.warning("settings.test_command %r is empty", command)
            return f"$ {command}\nFAILED TO START: empty test command"

        try:
            output, _code, timed_out = await self._run_command(tuple(argv), deadline)
        except OSError as exc:
            logger.warning("settings.test_command %r could not start: %s", command, exc)
            return f"$ {command}\nFAILED TO START: {exc}"
        if timed_out:
            return output
        return output + await self._run_static_checks(deadline)

    async def _autodetect(self, deadline: float) -> str:
        skipped: list[str] = []
        started = 0
        for candidate in _TEST_CANDIDATES:
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning("test timeout budget exhausted after %s", _headline(skipped))
                break
            try:
                output, _code, timed_out = await self._run_command(candidate, deadline)
            except OSError as exc:
                logger.debug("%s unavailable: %s", _display(candidate), exc)
                skipped.append(f"{_display(candidate)}: not available ({exc})")
                continue
            started += 1
            if timed_out:
                # The timeout is itself the finding; re-running the same suite
                # under another binary would just burn the rest of the budget.
                return output
            reason = _inconclusive_reason(output)
            if reason is not None:
                logger.debug("%s: %s", _display(candidate), reason)
                skipped.append(f"{_display(candidate)}: {reason}, trying next")
                continue
            return output + await self._run_static_checks(deadline)

        if not started:
            return NO_TEST_RUNNER
        return NO_USABLE_TEST_RUNNER + " Attempts:\n" + "\n".join(skipped)

    async def _run_static_checks(self, deadline: float) -> str:
        sections: list[str] = []
        for argv in _STATIC_CHECKS:
            try:
                output, _code, _timed_out = await self._run_command(argv, deadline)
            except OSError as exc:
                logger.debug("%s unavailable, skipping: %s", argv[0], exc)
                continue
            sections.append(output)
        if not sections:
            return ""
        return "\n\n" + "\n\n".join(sections)

    async def _run_command(
        self, argv: tuple[str, ...], deadline: float
    ) -> tuple[str, int, bool]:
        """Run ``argv`` in the workspace; return (output, returncode, timed_out).

        Raises :class:`OSError` when the binary cannot be executed at all.
        """
        timeout = max(1.0, deadline - asyncio.get_running_loop().time())
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.settings.workspace),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("%s timed out after %.0fs, killing it", _display(argv), timeout)
            with suppress(ProcessLookupError):
                proc.kill()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5)
            message = (
                f"TIMEOUT: exceeded the {self.settings.test_timeout}s test timeout, process killed."
            )
            return f"$ {_display(argv)}\nexit code: timeout\n{message}", -1, True

        code = proc.returncode if proc.returncode is not None else -1
        output = _decode(stdout) + _decode(stderr)
        rendered = f"$ {_display(argv)}\nexit code: {code}\n{_bound(output)}"
        return rendered, code, False

    def _build_prompt(self, goal: str, tasks: list[TaskModel], test_output: str) -> str:
        parts = [TESTER_PROMPT, f"Goal: {goal}", f"Tasks attempted this cycle: {len(tasks)}"]
        if tasks:
            listed = "\n".join(
                f"- {_status_text(task.status)} {task.id}: {excerpt(task.description, 160)}"
                for task in tasks[:_MAX_TASK_LINES]
            )
            parts.append(f"Tasks:\n{listed}")
        parts.append(f"Command output:\n{test_output}")
        parts.append("Return the JSON verdict now, grounded in the output above.")
        return "\n\n".join(parts)

    def _verdict(self, response: str) -> TestResult:
        data: Any = response
        try:
            data = parse_json_response(response)
        except ValueError:
            logger.debug("tester response is not JSON, scoring it as prose")
        try:
            result = coerce_test_result(data)
        except ValueError:
            logger.debug("tester JSON carries no usable verdict, scoring it as prose")
            return self._text_verdict(response)
        if not result.passed and not result.failures:
            # A failure with nothing to act on cannot drive the next cycle.
            result.failures = _failure_lines(result.summary or response)
        return result

    def _text_verdict(self, response: str) -> TestResult:
        text = response.strip()
        if not text:
            logger.warning("tester returned an empty verdict")
            return TestResult(
                passed=False,
                summary="Tester returned no verdict.",
                failures=["Tester returned an empty response."],
            )
        if looks_like_failure(text):
            return TestResult(passed=False, summary=text, failures=_failure_lines(text))
        if looks_like_pass(text):
            return TestResult(passed=True, summary=text, failures=[])
        return TestResult(
            passed=False,
            summary=text,
            failures=[f"Tester response was ambiguous: {excerpt(text)}"],
        )


def _status_text(status: object) -> str:
    return str(getattr(status, "value", status))


def _display(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def _headline(lines: Sequence[str]) -> str:
    return lines[0].splitlines()[0] if lines and lines[0].splitlines() else ""


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _bound(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the tail of a long log, where tracebacks and summaries live."""
    if len(text) <= limit:
        return text
    return f"...[output truncated to the last {limit} characters]...\n{text[-limit:]}"


def _inconclusive_reason(output: str) -> str | None:
    lowered = output.lower()
    for marker, reason in _INCONCLUSIVE_MARKERS:
        if marker in lowered:
            return reason
    return None


def _failure_lines(text: str) -> list[str]:
    lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
    if not lines:
        return [excerpt(text)]
    return [excerpt(line, 200) for line in lines[:_MAX_FAILURE_LINES]]
