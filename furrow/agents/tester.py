from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from furrow.config import Settings

# --------------------------------------------------------------------------- #
# Public sentinels / knobs
# --------------------------------------------------------------------------- #

#: ``TesterAgent.last_kind`` value meaning "no test tooling exists here".  A
#: project without a runner is *unverified*, not *broken*: reporting a failure
#: would make the orchestrator rewrite the goal to "fix failing tests" forever,
#: which is how the old "No test runner found." string became a permanent fake
#: failure.  Callers can branch on this via ``TesterAgent.no_test_runner``.
NO_TEST_RUNNER_KIND = "no_test_runner"
#: A real runner/static check executed and produced evidence.
TEST_RESULT_KIND_RAN = "ran"
#: The tester itself broke (spawn error, LLM outage); distinct from "tests failed".
TEST_RESULT_KIND_ERROR = "error"

#: Summary used for the vacuous pass.  Keep it free of failure vocabulary so the
#: orchestrator/LLM cannot read it as a fix request.  The ``[no-test-runner]``
#: marker makes the state recoverable from the summary string alone.
NO_RUNNER_SUMMARY = "[no-test-runner] No test runner detected; nothing to verify"
#: Used when the reply carries no usable verdict at all.
NO_PARSE_SUMMARY = "[unparseable-verdict] tester reply was not valid JSON"

#: Fallback used when ``settings.test_timeout`` is missing (older Settings).
DEFAULT_TEST_TIMEOUT = 120
#: Total wall-clock allowed for the *test* phase.  The old code gave every one of
#: seven candidate commands its own 120s budget, so a missing runner (or an npm
#: shim that installs forever) could stall the loop for minutes.  With a budget
#: the phase is capped regardless of how many candidates exist.
TEST_PHASE_BUDGET_FACTOR = 2
#: Same idea for the static-check phase, which is additive on top of the tests.
STATIC_CHECK_BUDGET_FACTOR = 1
#: Hard cap on any single captured stream, so multi-MB pytest output cannot be
#: pasted into an LLM prompt.
MAX_OUTPUT_CHARS = 4000
MAX_GOAL_CHARS = 4000
MAX_TASK_RESULT_CHARS = 600
MAX_PROMPT_TASKS = 20
TRUNCATION_MARKER = "... [truncated]"

#: pytest's exit code for "collected no tests" - vacuous, not a failure.
NO_TESTS_COLLECTED_EXIT = 5

#: Child processes get a clean env: enough to find binaries and a HOME, but no
#: ``*_API_KEY``/token leakage into third-party test runners.
_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TMP",
        "TEMP",
        "TZ",
        "PYTHONPATH",
        "PYTHONHASHSEED",
        "VIRTUAL_ENV",
        "PATHEXT",
        "SYSTEMROOT",
        "COMSPEC",
    }
)

_PYTHON_MARKERS = ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "setup.py")
_PYTHON_DIRS = ("tests", "test")
_NODE_FAMILIES = (
    # (binary, marker files, argv)
    ("pnpm", ("package.json", "pnpm-lock.yaml"), ["test", "--", "--silent"]),
    ("yarn", ("package.json", "yarn.lock"), ["test", "--silent"]),
    ("npm", ("package.json",), ["test", "--", "--silent"]),
)
_RUST_FAMILY = ("cargo", ("Cargo.toml",), ["test", "-q"])
_GO_FAMILY = ("go", ("go.mod",), ["test", "./..."])
#: Every non-python runner family, in probe order (pnpm, yarn, npm, cargo, go).
_FAMILIES = (*_NODE_FAMILIES, _RUST_FAMILY, _GO_FAMILY)
_STATIC_CHECKS = (("ruff", ["check", "."]), ("mypy", ["."]))

_EVIDENCE_HEADER = (
    "Evidence below was produced by running the project's own tooling. Exit codes "
    "are objective: treat a non-zero exit (or a TIMEOUT) as a failed run and do "
    "not report passed=true for it. pytest exit code 5 means 'no tests collected' "
    "and is NOT a failure."
)
_VERDICT_RULES = """Rules:
- Base the verdict on the evidence above, not on assumptions about the code.
- A non-zero exit code or a TIMEOUT from a test runner or linter means passed=false.
- If the evidence shows everything exited 0, passed=true unless the output shows
  something contradictory.
- Keep summary under one sentence and list one entry per distinct failure."""

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Bound ``text`` so a runaway log can never flood the prompt or the summary."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n{TRUNCATION_MARKER} ({dropped} chars omitted)"


def _decode(raw: bytes | None) -> str:
    return (raw or b"").decode("utf-8", errors="replace")


def _module_available(name: str) -> bool:
    """True when ``name`` can be imported, i.e. ``python -m <name>`` can work.

    Without this probe the ``python -m pytest`` fallback would emit
    "No module named pytest" and exit 1 - a fabricated failure.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def build_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Project a minimal environment for child processes (allowlist, not denylist)."""
    source = os.environ if base is None else base
    return {key: value for key, value in source.items() if key in _ENV_ALLOWLIST}


def _process_group_kwargs() -> dict[str, bool]:
    """``start_new_session`` puts the child in its own process group (POSIX only).

    That is what makes ``os.killpg`` able to reap a runner that spawned children
    of its own (pytest-xdist, npm, cargo); otherwise the timeout path would
    leave orphans behind.
    """
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


# --------------------------------------------------------------------------- #
# Runner detection
# --------------------------------------------------------------------------- #


def detect_test_commands(
    *,
    which: Callable[[str], str | None] | None = None,
    python: str | None = None,
    module_available: Callable[[str], bool] | None = None,
    exists: Callable[[Path], bool] | None = None,
    workspace: Path | None = None,
) -> list[list[str]]:
    """Return the ordered test commands to run for this project.

    Detection is a pure ``shutil.which`` probe - nothing is executed to find out
    whether a runner exists.  Running candidates to see what happens was the
    source of both the multi-minute stalls and the "first binary that happens to
    exist wins, even if it belongs to another ecosystem" bug.

    A family is only considered when its binary (or, for pytest, an importable
    module for ``sys.executable``) is available *and* the workspace looks like
    that ecosystem, so ``npm`` on PATH cannot hijack a Python project.

    Returns an empty list when nothing is available; the caller turns that into
    the vacuous "nothing to verify" pass instead of a failure.
    """
    which = which or shutil.which
    python = python or sys.executable
    module_available = module_available or _module_available
    exists = exists or Path.exists
    root = Path(workspace) if workspace is not None else Path.cwd()

    def present(*names: str) -> bool:
        return any(exists(root / name) for name in names)

    commands: list[list[str]] = []
    matched_marker = False

    python_ready = bool(which("pytest")) or (bool(python) and module_available("pytest"))
    if python_ready:
        python_marked = present(*_PYTHON_MARKERS) or present(*_PYTHON_DIRS)
        matched_marker = matched_marker or python_marked
        if python_marked:
            # Bare name (not the resolved path): PATH is forwarded to the child,
            # and bare commands keep the evidence table readable.
            if which("pytest"):
                commands.append(["pytest", "-q"])
            else:
                commands.append([python, "-m", "pytest", "-q"])

    for binary, markers, args in _FAMILIES:
        if not which(binary):
            continue
        marked = present(*markers)
        matched_marker = matched_marker or marked
        if marked:
            commands.append([binary, *args])

    if not matched_marker and not commands:
        # No project marker matched (exotic layout): fall back to "whatever
        # tooling is installed" rather than silently skipping verification.  The
        # argv is the family's own, so it is reused instead of re-declared here.
        for binary, args in (("pytest", ["-q"]), *((b, a) for b, _m, a in _FAMILIES)):
            if which(binary):
                commands.append([binary, *args])
        if not commands and python and module_available("pytest"):
            commands.append([python, "-m", "pytest", "-q"])

    return commands


def detect_static_checks(
    *,
    which: Callable[[str], str | None] | None = None,
    python: str | None = None,
    module_available: Callable[[str], bool] | None = None,
) -> list[list[str]]:
    """Lint/type commands to run, skipping tools that are not installed.

    Missing ``mypy``/``ruff`` must not produce a failure line, otherwise every
    environment without dev extras reports a permanent "fix failures" goal.
    """
    which = which or shutil.which
    python = python or sys.executable
    module_available = module_available or _module_available
    checks: list[list[str]] = []
    for tool, args in _STATIC_CHECKS:
        if which(tool):
            checks.append([tool, *args])
        elif python and module_available(tool):
            checks.append([python, "-m", tool, *args])
    return checks


# --------------------------------------------------------------------------- #
# Command execution
# --------------------------------------------------------------------------- #


def _kill_process_group(proc: object) -> str | None:
    """Kill the runner and anything it spawned; return how it was killed."""
    if getattr(proc, "returncode", None) is not None:
        return None
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # type: ignore[attr-defined]
            return "killpg"
        except OSError:
            pass
    try:
        proc.kill()  # type: ignore[attr-defined]
        return "kill"
    except OSError:
        return None


async def _reap(proc: object) -> None:
    """Wait for a killed child so it does not linger as a zombie.

    ``CancelledError`` is deliberately *not* swallowed: a shutdown that arrives
    while we are reaping must still unwind the whole tester run.
    """
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)  # type: ignore[attr-defined]
    except asyncio.TimeoutError:
        pass


async def run_command(
    argv: list[str],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TEST_TIMEOUT,
) -> CommandOutcome:
    """Run ``argv`` with a hard timeout and capture argv/exit code/output."""
    outcome = CommandOutcome(argv=list(argv), label=argv[0] if argv else "", timeout=timeout)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_process_group_kwargs(),
        )
    except (OSError, ValueError) as exc:
        # Covers FileNotFoundError plus "binary is not runnable"; never swallow
        # anything broader, otherwise real bugs look like "no runner".
        outcome.spawn_error = f"{type(exc).__name__}: {exc}"
        return outcome

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        outcome.timed_out = True
        outcome.kill_method = _kill_process_group(proc)
        await _reap(proc)
    except asyncio.CancelledError:
        # Our own cancellation: kill synchronously and let the error propagate.
        _kill_process_group(proc)
        raise
    else:
        outcome.stdout = _truncate(_decode(stdout))
        outcome.stderr = _truncate(_decode(stderr))

    outcome.exit_code = proc.returncode
    return outcome


# --------------------------------------------------------------------------- #
# Evidence model
# --------------------------------------------------------------------------- #


@dataclass
class CommandOutcome:
    """Objective record of one executed command."""

    argv: list[str]
    label: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timeout: float | None = None
    timed_out: bool = False
    spawn_error: str | None = None
    skip_reason: str | None = None
    kill_method: str | None = None

    @property
    def command_line(self) -> str:
        return " ".join(self.argv)

    @property
    def failed(self) -> bool:
        """Non-zero exit (or an unrunnable/timed-out run) is failure evidence."""
        if self.skip_reason is not None:
            return False
        if self.spawn_error or self.timed_out:
            return True
        if self.exit_code is None:
            return False
        if self.exit_code == NO_TESTS_COLLECTED_EXIT and self.label.startswith("pytest"):
            return False
        return self.exit_code != 0

    @property
    def status(self) -> str:
        if self.skip_reason is not None:
            return f"SKIPPED ({self.skip_reason})"
        if self.spawn_error is not None:
            return f"SPAWN ERROR ({self.spawn_error})"
        if self.timed_out:
            timeout = f"{self.timeout:g}s" if self.timeout else "timeout"
            killed = f", killed via {self.kill_method}" if self.kill_method else ""
            return f"TIMEOUT after {timeout}{killed}"
        if self.exit_code is None:
            return "NO EXIT CODE"
        if self.exit_code == NO_TESTS_COLLECTED_EXIT and self.label.startswith("pytest"):
            return f"exit {self.exit_code} (no tests collected - vacuous, not a failure)"
        return f"exit {self.exit_code}"

    def failure_line(self) -> str:
        """One bounded line for ``TestResult.failures`` (tool name always first)."""
        if self.timed_out:
            timeout = f"{self.timeout:g}s" if self.timeout else "timeout"
            return f"{self.label}: TIMEOUT after {timeout}"
        if self.spawn_error is not None:
            return f"{self.label}: could not be executed ({self.spawn_error})"
        tail = _last_meaningful_line(f"{self.stdout}\n{self.stderr}")
        base = f"{self.label}: exit {self.exit_code}"
        return f"{base} -- {tail}" if tail else base

    def render(self) -> str:
        lines = [f"- `{self.command_line}` -> {self.status}"]
        for stream, text in (("stdout", self.stdout), ("stderr", self.stderr)):
            if text.strip():
                lines.append(f"  {stream}:")
                lines.extend(f"    {line}" for line in text.splitlines())
        if self.skip_reason is not None:
            lines.append("  (no evidence: this command was not executed)")
        return "\n".join(lines)


@dataclass
class VerificationReport:
    """Everything the tester observed, before any LLM interpretation."""

    commands: list[CommandOutcome] = field(default_factory=list)
    static_checks: list[CommandOutcome] = field(default_factory=list)
    no_runner: bool = False

    @property
    def executed(self) -> list[CommandOutcome]:
        return [*self.commands, *self.static_checks]

    def failure_lines(self) -> list[str]:
        return [outcome.failure_line() for outcome in self.executed if outcome.failed]

    def digest(self) -> str:
        """Compact one-line-per-command digest for ``TestResult.summary``."""
        if self.no_runner:
            return NO_RUNNER_SUMMARY
        parts = [
            f"{outcome.label}: {_truncate(outcome.status, 120)}"
            for outcome in self.executed
            if outcome.skip_reason is None
        ]
        if not parts:
            parts = ["no command produced evidence (time budget exhausted)"]
        return "; ".join(parts)

    def render(self) -> str:
        sections = [
            "Test commands:",
            *([o.render() for o in self.commands] or ["- (none detected)"]),
            "Lint / type checks:",
            *([o.render() for o in self.static_checks] or ["- (no linter installed)"]),
        ]
        return "\n".join(sections)


def _last_meaningful_line(text: str, limit: int = 200) -> str:
    for line in reversed([line.strip() for line in text.splitlines()]):
        if line:
            return _truncate(line, limit)
    return ""


# --------------------------------------------------------------------------- #
# LLM response parsing
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"```(?:[A-Za-z0-9_+-]*)?\s*(.*?)```", re.DOTALL)
_PASSED_MARKER = re.compile(r'"passed"\s*:\s*(true|false)', re.IGNORECASE)
_COUNT = re.compile(r"(\d+)\s+(passed|failed)\b", re.IGNORECASE)
_PASSED_TOKEN = re.compile(r"\bpassed\b", re.IGNORECASE)
_FAILED_TOKEN = re.compile(r"\bfailed\b", re.IGNORECASE)
_NEGATED_PASS = re.compile(r"\b(?:not|un|non|never)[\s-]?(?:passed|passing|pass)\b", re.IGNORECASE)


def extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of a reply, tolerating ```json fences."""
    candidates = [blob.strip() for blob in _FENCE.findall(text)]
    candidates.append(text)
    for blob in candidates:
        start = blob.find("{")
        if start == -1:
            continue
        try:
            data, _ = json.JSONDecoder().raw_decode(blob[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def heuristic_passed(text: str) -> bool | None:
    """Best-effort verdict when the reply is not valid JSON.

    Deliberately *not* ``"passed" in text.lower()``: that returns True for
    "not passed", "unpassed" and "0 passed, 1 failed".  Order of trust:
    explicit ``"passed": true|false`` > ``N passed``/``N failed`` counts >
    standalone status tokens after stripping negated-pass phrases.  ``None``
    means "no signal"; the caller then falls back to exit codes.
    """
    markers = _PASSED_MARKER.findall(text)
    if markers:
        return markers[-1].lower() == "true"

    counts = _COUNT.findall(text)
    passed_n = sum(int(n) for n, word in counts if word.lower() == "passed")
    failed_n = sum(int(n) for n, word in counts if word.lower() == "failed")
    if passed_n or failed_n:
        return failed_n == 0 and passed_n > 0

    cleaned = _NEGATED_PASS.sub(" ", text)
    if _FAILED_TOKEN.search(cleaned):
        return False
    if _PASSED_TOKEN.search(cleaned):
        return True
    if _NEGATED_PASS.search(text):
        # "not passed" / "unpassed" states a failure; it must never read as a pass.
        return False
    return None


def parse_llm_result(response: str) -> TestResult | None:
    """Structured verdict from a (possibly fenced) JSON reply, else ``None``."""
    data = extract_json_object(response)
    if not isinstance(data, dict) or not isinstance(data.get("passed"), bool):
        return None
    summary = data.get("summary")
    failures = data.get("failures")
    return TestResult(
        passed=data["passed"],
        summary=_truncate(str(summary), 600) if summary else NO_PARSE_SUMMARY,
        failures=[_truncate(str(item), 400) for item in failures]
        if isinstance(failures, list)
        else [],
    )


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class TesterAgent:
    """Verification gate: run the project's own tooling, then have the LLM judge it.

    Public API: ``await TesterAgent(client=..., settings=...).run(goal, tasks)``.
    """

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or getattr(self.client, "settings", None) or default_settings
        #: One of ``NO_TEST_RUNNER_KIND`` / ``TEST_RESULT_KIND_RAN`` /
        #: ``TEST_RESULT_KIND_ERROR`` after each :meth:`run`.
        self.last_kind: str | None = None

    # -- introspection ----------------------------------------------------- #

    @property
    def no_test_runner(self) -> bool:
        """True when the last :meth:`run` found no runner (vacuous pass).

        This is how callers tell "nothing to verify" from "tests failed" without
        parsing prose: ``TestResult`` has no such field and is owned elsewhere.
        """
        return self.last_kind == NO_TEST_RUNNER_KIND

    @property
    def test_timeout(self) -> float:
        """Per-command wall-clock budget; ``settings.test_timeout`` when available."""
        raw = getattr(self.settings, "test_timeout", DEFAULT_TEST_TIMEOUT)
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            value = DEFAULT_TEST_TIMEOUT
        return value if value > 0 else float(DEFAULT_TEST_TIMEOUT)

    @property
    def workspace(self) -> Path:
        return Path(getattr(self.settings, "workspace", Path.cwd()))

    # -- main entry point -------------------------------------------------- #

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            report = await self._run_tests()
        except asyncio.CancelledError:
            raise  # shutdown must not be reported as a test failure
        except Exception as exc:
            return self._error(f"verification could not run ({type(exc).__name__}: {exc})")

        if report.no_runner:
            self.last_kind = NO_TEST_RUNNER_KIND
            return TestResult(passed=True, summary=NO_RUNNER_SUMMARY, failures=[])

        prompt = self._build_prompt(goal, tasks, report)
        try:
            response = await self.client.complete(prompt, model=self._tester_model())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # The tooling already produced evidence; a provider outage must not
            # fabricate a test failure (that is how the loop gets stuck).
            self.last_kind = TEST_RESULT_KIND_ERROR
            summary = f"[llm-unavailable] {type(exc).__name__}: {exc}"
            return self._from_evidence(report, None, summary_prefix=summary)

        self.last_kind = TEST_RESULT_KIND_RAN
        parsed = parse_llm_result(response)
        if parsed is not None:
            return self._from_evidence(report, parsed)
        return self._from_evidence(
            report, None, summary_prefix=f"{NO_PARSE_SUMMARY}; reply={_truncate(response.strip(), 300)!r}"
        )

    # -- execution --------------------------------------------------------- #

    async def _run_tests(self) -> VerificationReport:
        """Execute the detected runners plus lint checks and collect evidence."""
        commands = detect_test_commands(workspace=self.workspace, python=sys.executable)
        if not commands:
            return VerificationReport(no_runner=True)

        env = build_env()
        cwd = self.workspace
        outcomes = await self._run_phase(
            commands,
            cwd=cwd,
            env=env,
            budget_factor=TEST_PHASE_BUDGET_FACTOR,
            exhausted="test phase time budget exhausted",
        )
        static_checks = await self._run_static_checks(cwd=cwd, env=env)
        return VerificationReport(commands=outcomes, static_checks=static_checks)

    async def _run_static_checks(
        self, *, cwd: Path, env: Mapping[str, str]
    ) -> list[CommandOutcome]:
        return await self._run_phase(
            detect_static_checks(python=sys.executable),
            cwd=cwd,
            env=env,
            budget_factor=STATIC_CHECK_BUDGET_FACTOR,
            exhausted="static-check budget exhausted",
        )

    async def _run_phase(
        self,
        commands: list[list[str]],
        *,
        cwd: Path,
        env: Mapping[str, str],
        budget_factor: float,
        exhausted: str,
    ) -> list[CommandOutcome]:
        """Run ``commands`` one after another under a shared wall-clock budget.

        Commands are skipped (never run without a budget) once the phase budget is
        gone; ``exhausted`` is recorded as the reason so the prompt can say the
        command produced no evidence instead of implying it passed.
        """
        timeout = self.test_timeout
        deadline = time.monotonic() + timeout * budget_factor
        outcomes: list[CommandOutcome] = []
        for argv in commands:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                outcomes.append(
                    CommandOutcome(argv=list(argv), label=argv[0], skip_reason=exhausted)
                )
                continue
            outcome = await run_command(argv, cwd=cwd, env=env, timeout=min(timeout, remaining))
            outcome.label = _label_for(argv)
            outcomes.append(outcome)
        return outcomes

    # -- prompting --------------------------------------------------------- #

    def _build_prompt(self, goal: str, tasks: list[TaskModel], report: VerificationReport) -> str:
        sections = [
            TESTER_PROMPT,
            "",
            f"Goal: {_truncate(goal, MAX_GOAL_CHARS)}",
            "",
            "Task results produced in this cycle (does the implementation match them?):",
            self._render_tasks(tasks),
            "",
            _EVIDENCE_HEADER,
            "",
            report.render(),
            "",
            _VERDICT_RULES,
        ]
        return "\n".join(sections)

    @staticmethod
    def _render_tasks(tasks: list[TaskModel]) -> str:
        if not tasks:
            return "- (no task results available)"
        lines = []
        for task in tasks[:MAX_PROMPT_TASKS]:
            result = _truncate(task.result or "-", MAX_TASK_RESULT_CHARS).replace("\n", " ")
            lines.append(f"- [{task.id}] {task.status}: {task.description} -> {result}")
        if len(tasks) > MAX_PROMPT_TASKS:
            lines.append(f"- ... {len(tasks) - MAX_PROMPT_TASKS} more task(s) omitted")
        return "\n".join(lines)

    # -- verdict ----------------------------------------------------------- #

    def _error(self, message: str) -> TestResult:
        self.last_kind = TEST_RESULT_KIND_ERROR
        return TestResult(passed=False, summary=message, failures=[_truncate(message, 400)])

    def _from_evidence(
        self,
        report: VerificationReport,
        parsed: TestResult | None,
        *,
        summary_prefix: str | None = None,
    ) -> TestResult:
        """Combine the LLM verdict with exit codes; evidence always wins on failure."""
        failures = report.failure_lines()
        if failures:
            # A non-zero exit / timeout is objective: never let a "passed=true"
            # reply (or a missing reply) turn it into a green run.
            passed = False
            llm_failures = list(parsed.failures) if parsed else []
            if parsed is not None and parsed.passed:
                summary_prefix = "[overridden-by-exit-codes] LLM reported a pass"
            failures = [*failures, *[f for f in llm_failures if f not in failures]]
        elif parsed is not None:
            passed = parsed.passed
            # The LLM's own failure list is the only evidence there is when every
            # command exited 0; dropping it would hand the next cycle an empty
            # failure list to work from.
            failures = [] if parsed.passed else list(parsed.failures)
        else:
            passed = True  # every command exited 0 and the LLM gave no signal
            failures = []

        parts = [p for p in (summary_prefix, parsed.summary if parsed else None) if p]
        parts.append(f"[evidence] {report.digest()}")
        summary = _truncate("\n".join(parts), 1200)
        return TestResult(passed=passed, summary=summary, failures=failures[:20])

    def _tester_model(self) -> str | None:
        return getattr(self.settings, "tester_model", None)


def _label_for(argv: list[str]) -> str:
    """Short human label for evidence rows ("npm test", "ruff", "pytest")."""
    if not argv:
        return "<empty>"
    base = Path(argv[0]).name
    if base.startswith("python"):
        # "<python> -m pytest" -> "pytest", "<python> -m mypy ." -> "mypy"
        if len(argv) > 2 and argv[1] == "-m":
            return argv[2]
        return base
    if base in {"npm", "pnpm", "yarn", "cargo", "go"}:
        return f"{base} {argv[1]}" if len(argv) > 1 else base
    return base