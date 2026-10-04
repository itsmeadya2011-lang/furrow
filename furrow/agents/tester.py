"""Verification agent: runs the workspace's real test command and grades it.

Measurement and explanation are deliberately kept apart:

* **Measurement** is the subprocess exit code. ``passed`` is exactly
  ``returncode == 0``. It is never inferred from the text of the output -- a
  substring search for ``"passed"`` calls ``"0 passed, 3 failed"`` a pass.
* **Explanation** is the LLM's job. It summarises the captured output and lists
  the individual failures, but whatever it claims about ``passed`` is discarded,
  so a confident-but-wrong model can never turn a red suite green.

``TestResult.verified`` records whether a runner actually ran. It is ``False``
**only** when no test runner could be detected or every spawn attempt failed, in
which case ``passed`` carries no signal at all. A genuinely failing suite is
``verified=True, passed=False`` so the orchestrator can start a fix cycle; a
timed-out run is ``verified=True, passed=False`` too, because the runner did run
and did not succeed.

Optional lint and type checks live here because ``.kilo/agent/furrow-tester.md``
asks for tests *and* lint *and* type checks, while the orchestrator's hot path
only ever wants tests. Both default to off::

    result = await TesterAgent(client=client).run(
        goal, plan.tasks, run_lint=True, run_typecheck=True
    )

Lint/type checks are best-effort: a missing ``ruff``/``mypy`` is noted in the
summary, never fatal, and never touches ``verified``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from anthropic import APIError as AnthropicAPIError
from openai import OpenAIError

from furrow.agents.planner import extract_json_object
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

SYSTEM_PROMPT = (
    "You are a test-output summariser for an autonomous coding system. "
    "Respond with raw JSON only - no markdown fences, no prose."
)

MAX_ATTEMPTS = 3
DEFAULT_TIMEOUT = 120.0
REAP_TIMEOUT = 5.0
DRAIN_TIMEOUT = 1.0
MAX_OUTPUT_CHARS = 20_000
MAX_TRANSCRIPT_CHARS = 2 * MAX_OUTPUT_CHARS
MAX_FAILURES = 20
MAX_FAILURE_CHARS = 600
MAX_SUMMARY_CHARS = 2400

_PYTHON_MARKERS = ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini")
_NODE_MARKERS = ("pnpm-lock.yaml", "yarn.lock")
_ALL_MARKERS = (*_PYTHON_MARKERS, "package.json", *_NODE_MARKERS, "Cargo.toml", "go.mod")
_FURROW_PREFIX = "FURROW_"
_CREDENTIAL_ENV = frozenset({"ANTHROPIC_API_KEY", "OPENAI_API_KEY"})

# Ways the optional explanation step can fail without touching the verdict:
# provider transport errors, a missing API key (ValueError), socket-level trouble
# (OSError) and a client-side timeout.
_LLM_ERRORS: tuple[type[BaseException], ...] = (
    AnthropicAPIError,
    OpenAIError,
    OSError,
    asyncio.TimeoutError,
    ValueError,
)


@dataclass(frozen=True)
class CommandResult:
    """Structured outcome of one command execution."""

    command: tuple[str, ...]
    returncode: int | None
    output: str
    timed_out: bool = False
    spawn_failed: bool = False

    @property
    def command_line(self) -> str:
        return " ".join(self.command)


@dataclass(frozen=True)
class Runner:
    """A resolved command plus the marker file (or PATH lookup) that chose it."""

    ecosystem: str
    command: tuple[str, ...]
    marker: str
    fallback: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Explanation:
    """LLM prose about a run. ``summary is None`` means "fall back to the exit code"."""

    summary: str | None = None
    failures: tuple[str, ...] = ()
    error: str | None = None


LINT_RUNNER = Runner("lint", ("ruff", "check", "."), "ruff on PATH")
TYPECHECK_RUNNER = Runner("typecheck", ("mypy", "."), "mypy on PATH")


def scrubbed_env() -> dict[str, str]:
    """Copy the parent environment minus the agent's own credentials.

    Test code -- and anything it imports, or any code it downloads at import
    time -- must never be able to read the keys that pay for the LLM calls, so
    the known provider keys and every ``FURROW_*`` setting are dropped.
    """
    env = dict(os.environ)
    for key in list(env):
        if key in _CREDENTIAL_ENV or key.startswith(_FURROW_PREFIX):
            del env[key]
    return env


def bounded_output(text: str) -> str:
    """Clamp captured output to ``MAX_OUTPUT_CHARS``, keeping the informative tail."""
    if not text.strip():
        return "(no output)"
    if len(text) > MAX_OUTPUT_CHARS:
        dropped = len(text) - MAX_OUTPUT_CHARS
        head = f"[... {dropped} earlier characters truncated ...]\n"
        return head + text[-MAX_OUTPUT_CHARS:]
    return text


def format_output(stdout: str, stderr: str) -> str:
    """Label stdout and stderr separately instead of gluing them into one blob."""
    sections: list[str] = []
    if stdout.strip():
        sections.append(f"--- stdout ---\n{stdout.rstrip()}")
    if stderr.strip():
        sections.append(f"--- stderr ---\n{stderr.rstrip()}")
    return bounded_output("\n".join(sections))


def tail(text: str, limit: int = 1500) -> str:
    """Last ``limit`` characters of ``text`` -- where pytest/jest print the reasons."""
    text = text.rstrip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> str:
    """Kill a hung child, reap it, and salvage whatever output it already wrote.

    ``wait()`` after ``kill()`` is what turns the child from a zombie into a
    reaped process; without it a hung suite leaves orphans behind. The drain is
    best-effort and bounded, because the whole point of this path is that we
    cannot afford to wait any longer.
    """
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=REAP_TIMEOUT)

    chunks: list[str] = []
    for label, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        if stream is None:
            continue
        with contextlib.suppress(asyncio.TimeoutError, OSError, ValueError):
            data = await asyncio.wait_for(stream.read(), timeout=DRAIN_TIMEOUT)
            if data:
                chunks.append(f"--- {label} (partial) ---\n{data.decode(errors='replace')}")
    return bounded_output("\n".join(chunks))


async def run_command(
    command: tuple[str, ...],
    *,
    cwd: str,
    timeout: float | None = None,
) -> CommandResult:
    """Run one command in ``cwd`` with a scrubbed environment and a hard timeout.

    The expected failure modes come back as flags instead of exceptions: a missing
    executable is ``spawn_failed``, a hang is ``timed_out``. Everything else is a
    real bug and is allowed to propagate.

    ``timeout`` defaults to the module-level ``DEFAULT_TIMEOUT``, read at call
    time so tests can shorten it without reimporting the module.
    """
    limit = DEFAULT_TIMEOUT if timeout is None else timeout
    line = " ".join(command)
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=scrubbed_env(),
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        return CommandResult(command, None, f"could not start `{line}`: {e}", spawn_failed=True)

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=limit)
    except asyncio.TimeoutError:
        # A timeout is a failure, not an excuse to try the next runner: the single
        # detected runner is the right one, and it did not finish.
        return CommandResult(command, None, await _kill_and_reap(proc), timed_out=True)
    return CommandResult(
        command,
        proc.returncode,
        format_output(stdout.decode(errors="replace"), stderr.decode(errors="replace")),
    )


def detect_runner(workspace: Path) -> Runner | None:
    """Pick the one runner that matches this workspace, or ``None`` if there is none.

    Detection replaces the old "try seven commands in order" approach, which let a
    Python repo be graded on ``go test ./...`` output.
    """
    for marker in _PYTHON_MARKERS:
        if (workspace / marker).exists():
            return Runner(
                "python",
                ("python", "-m", "pytest", "-q"),
                marker,
                fallback=("pytest", "-q"),
            )
    if (workspace / "package.json").exists():
        if (workspace / "pnpm-lock.yaml").exists():
            return Runner("node", ("pnpm", "test", "--", "--silent"), "pnpm-lock.yaml")
        if (workspace / "yarn.lock").exists():
            return Runner("node", ("yarn", "test", "--silent"), "yarn.lock")
        return Runner("node", ("npm", "test", "--", "--silent"), "package.json")
    if (workspace / "Cargo.toml").exists():
        return Runner("rust", ("cargo", "test", "-q"), "Cargo.toml")
    if (workspace / "go.mod").exists():
        return Runner("go", ("go", "test", "./..."), "go.mod")
    return None


def _no_runner_summary(workspace: Path) -> str:
    """Diagnostic naming exactly what was looked for, and finding nothing."""
    if not workspace.is_dir():
        where = f"workspace {workspace} does not exist"
    else:
        where = f"workspace {workspace} contains none of {', '.join(_ALL_MARKERS)}"
    return (
        f"No test runner detected: {where}. "
        "Tests were never executed, so this result carries no pass/fail signal "
        "(verified=False)."
    )


def _spawn_failure_summary(runner: Runner, attempts: list[CommandResult]) -> str:
    """Diagnostic for "the right runner was detected but nothing could be started"."""
    tried = ", ".join(f"`{a.command_line}`" for a in attempts)
    why = "; ".join(tail(a.output, 200) for a in attempts)
    return (
        f"No usable test runner for the detected {runner.ecosystem} project "
        f"(looked for {runner.marker}): none of {tried} could be started ({why}). "
        "Tests were never executed, so this result carries no pass/fail signal "
        "(verified=False)."
    )


def _build_result(
    passed: bool,
    summary: str,
    failures: list[str],
    *,
    verified: bool,
) -> TestResult:
    """Construct a TestResult, tolerating configs that predate the ``verified`` field."""
    if "verified" in TestResult.model_fields:
        return TestResult(passed=passed, summary=summary, failures=failures, verified=verified)
    return TestResult(passed=passed, summary=summary, failures=failures)


def _explain_from(data: dict) -> Explanation:
    """Read the model's prose, ignoring its opinion about ``passed`` and ``verified``."""
    raw_summary = data.get("summary")
    summary = raw_summary.strip() if isinstance(raw_summary, str) and raw_summary.strip() else None
    raw_failures = data.get("failures")
    failures: list[str] = []
    if isinstance(raw_failures, list):
        for item in raw_failures[:MAX_FAILURES]:
            text = item.strip() if isinstance(item, str) else str(item).strip()
            if text:
                failures.append(text[:MAX_FAILURE_CHARS])
    return Explanation(summary=summary, failures=tuple(failures))


class TesterAgent:
    """Runs the workspace's test command and turns its exit code into a TestResult."""

    # Not a pytest class: the "Test" prefix would otherwise make pytest try to
    # collect this class as a test case. pytest honours this attribute.
    __test__ = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(
        self,
        goal: str,
        tasks: list[TaskModel],
        *,
        run_lint: bool = False,
        run_typecheck: bool = False,
    ) -> TestResult:
        workspace = Path(self.client.settings.workspace)
        runner = detect_runner(workspace)
        if runner is None:
            return _build_result(False, _no_runner_summary(workspace), [], verified=False)

        attempts = await self._execute(runner, workspace)
        if all(attempt.spawn_failed for attempt in attempts):
            return _build_result(
                False, _spawn_failure_summary(runner, attempts), [], verified=False
            )
        # The last attempt is the one that ran: the primary command, or the
        # fallback when only the fallback could be started.
        outcome = attempts[-1]

        extras: list[tuple[str, CommandResult]] = []
        if run_lint:
            extras.append(("lint", (await self._execute(LINT_RUNNER, workspace))[-1]))
        if run_typecheck:
            extras.append(("typecheck", (await self._execute(TYPECHECK_RUNNER, workspace))[-1]))

        # ------------------------------------------------------------------
        # VERDICT PRECEDENCE: the exit code decides. It is computed before the
        # LLM is consulted and is overwritten by nothing -- the model's answer
        # is only allowed to contribute a summary and a failure list.
        # ------------------------------------------------------------------
        passed = outcome.returncode == 0 and not outcome.timed_out
        measured = _suite_failures(outcome)
        notes: list[str] = []
        for label, result in extras:
            if result.spawn_failed:
                notes.append(f"{label}: `{result.command_line}` unavailable, skipped")
            elif result.timed_out:
                passed = False
                measured.append(f"{label} timed out after {DEFAULT_TIMEOUT:.0f}s")
            elif result.returncode != 0:
                passed = False
                measured.append(
                    f"{label} reported errors (exit {result.returncode}):\n{tail(result.output)}"
                )

        explanation = await self._explain(
            self._prompt(
                goal,
                tasks,
                runner,
                outcome,
                workspace=workspace,
                skipped=attempts[:-1],
                extras=extras,
                passed=passed,
            )
        )
        if explanation.error:
            notes.append(f"LLM summary unavailable ({explanation.error})")

        summary = explanation.summary or _fallback_summary(runner, outcome)
        reported = list(explanation.failures)
        if passed and reported:
            # Exit code 0 means the suite passed; do not hand the orchestrator a
            # failure list for a green run, but say that the model disagreed.
            notes.append(f"model listed {len(reported)} failures despite exit 0 (ignored)")
        failures = (measured + ([] if passed else reported))[:MAX_FAILURES]
        if notes:
            summary = f"{summary} [{'; '.join(notes)}]"

        return _build_result(passed, summary.strip()[:MAX_SUMMARY_CHARS], failures, verified=True)

    async def _execute(self, runner: Runner, workspace: Path) -> list[CommandResult]:
        """Run a runner's primary command, plus its fallback only if the spawn failed."""
        results = [await run_command(runner.command, cwd=str(workspace))]
        if results[0].spawn_failed and runner.fallback is not None:
            results.append(await run_command(runner.fallback, cwd=str(workspace)))
        return results

    def _prompt(
        self,
        goal: str,
        tasks: list[TaskModel],
        runner: Runner,
        outcome: CommandResult,
        *,
        workspace: Path,
        skipped: list[CommandResult],
        extras: list[tuple[str, CommandResult]],
        passed: bool,
    ) -> str:
        """Build the judge prompt, including what this cycle actually touched."""
        touched = sorted({path for task in tasks for path in task.files})
        changed = "\n".join(f"- {path}" for path in touched) or "- (no files reported)"

        sections = [
            TESTER_PROMPT,
            f"Goal: {goal}",
            f"Project type: {runner.ecosystem} (detected from {runner.marker})",
            f"Working directory: {workspace}",
            f"Files touched in this cycle:\n{changed}",
        ]
        for attempt in skipped:
            sections.append(f"Skipped `{attempt.command_line}`: {attempt.output}")
        sections.append(
            f"Command: `{outcome.command_line}`\n"
            f"Exit code: {outcome.returncode if not outcome.timed_out else 'TIMED OUT'}\n"
            f"{outcome.output}"
        )
        for label, result in extras:
            state = (
                "not installed"
                if result.spawn_failed
                else ("timed out" if result.timed_out else f"exit {result.returncode}")
            )
            sections.append(f"{label} check (`{result.command_line}`): {state}\n{result.output}")

        verdict = "PASSED" if passed else "FAILED"
        sections.append(
            f"The verdict is already {verdict}: it is taken from the process result above "
            "(the exit code, or TIMED OUT) and is final. Your job is only to write the "
            "human-readable summary and the list of failures.\n"
            f'- Report "passed" as {"true" if passed else "false"} to match the verdict; never '
            "override it.\n"
            "- Make each entry in `failures` concrete: the failing test name and its error "
            "message.\n"
            "- If the command timed out or failed during collection, say that instead of "
            "inventing test names.\n"
            'Return JSON only: {"passed": '
            f'{str(passed).lower()}, "summary": "...", "failures": ["..."]}}'
        )
        return bounded_output("\n\n".join(sections))[:MAX_TRANSCRIPT_CHARS]

    async def _explain(self, prompt: str) -> Explanation:
        """Ask the LLM to describe the run, with bounded retries on unusable JSON.

        Mirrors the planner's retry pattern. Only prose comes back: a model that
        cannot produce parseable JSON (or cannot be reached at all) leaves the
        exit-code verdict untouched and simply yields a thinner summary.
        """
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await self.client.complete(
                    prompt,
                    system=SYSTEM_PROMPT,
                    model=self.client.settings.model_for("tester"),
                )
            except _LLM_ERRORS as e:
                # The verdict is already fixed by the exit code, so a transport
                # failure here loses prose, not correctness. It is reported in the
                # summary instead of being swallowed.
                return Explanation(error=f"{type(e).__name__}: {e}")

            try:
                return _explain_from(extract_json_object(response))
            except ValueError as e:
                if attempt == MAX_ATTEMPTS - 1:
                    return Explanation(error=str(e))
                snippet = response[:200]
                prompt = (
                    f"{prompt}\n\nYour previous response was invalid: {e}\n"
                    f"Malformed output:\n{snippet}\n\n"
                    'Respond with raw JSON only: {"passed": ..., "summary": "...", '
                    '"failures": ["..."]}.'
                )
        return Explanation(error="unreachable")


def _suite_failures(outcome: CommandResult) -> list[str]:
    """Failures the exit code proves on its own, independent of the model."""
    if outcome.timed_out:
        return [
            (
                f"`{outcome.command_line}` timed out after {DEFAULT_TIMEOUT:.0f}s: "
                f"{tail(outcome.output, 500)}"
            )
        ]
    if outcome.returncode == 0:
        return []
    return [f"`{outcome.command_line}` exited {outcome.returncode}:\n{tail(outcome.output)}"]


def _fallback_summary(runner: Runner, outcome: CommandResult) -> str:
    """A summary derived purely from the exit code, used when the LLM gives us nothing."""
    if outcome.timed_out:
        return (
            f"{runner.ecosystem} tests FAILED: `{outcome.command_line}` timed out after "
            f"{DEFAULT_TIMEOUT:.0f}s."
        )
    if outcome.returncode == 0:
        return f"{runner.ecosystem} tests PASSED: `{outcome.command_line}` exited 0."
    return f"{runner.ecosystem} tests FAILED: `{outcome.command_line}` exited {outcome.returncode}."
