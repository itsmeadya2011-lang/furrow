"""Tester agent: run the project's real checks and report only verified results.

Two invariants drive this module:

1. Every applicable check is run and recorded; the exit codes are the ground
   truth about the repository.
2. The LLM verdict is advisory.  A hallucinated ``"passed": true`` can never
   override a check that actually exited non-zero.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

from furrow.agents.json_parse import parse_json_object
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

#: Hard per-command wall-clock bound so one hung check cannot stall the loop.
CHECK_TIMEOUT_SECONDS: Final[float] = 60.0
#: Seconds to wait for a timed-out child to be reaped before giving up.
KILL_TIMEOUT_SECONDS: Final[float] = 5.0
#: Per-check captured output cap, keeping prompts bounded.
MAX_OUTPUT_CHARS: Final[int] = 8000
#: Exit code reported when a check exceeds ``CHECK_TIMEOUT_SECONDS``.
TIMEOUT_RETURNCODE: Final[int] = 124

PYTHON_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini")
NODE_MARKER: Final[str] = "package.json"
RUST_MARKER: Final[str] = "Cargo.toml"
GO_MARKER: Final[str] = "go.mod"

NO_CHECKS_MESSAGE: Final[str] = (
    "No checks were run: none of the supported project markers "
    f"({', '.join((*PYTHON_MARKERS, NODE_MARKER, RUST_MARKER, GO_MARKER))}) were found."
)


class CheckResult(NamedTuple):
    """Outcome of a single verification command."""

    name: str
    command: str
    returncode: int | None
    output: str
    skipped: bool = False

    @property
    def executed(self) -> bool:
        """True when the command actually ran (as opposed to being skipped)."""
        return not self.skipped


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    collapsed = text.strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + f"\n... [truncated {len(collapsed) - limit} chars]"


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _first_line(text: str, limit: int = 160) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:limit]
    return ""


def format_check_results(results: Sequence[CheckResult]) -> str:
    """Render ``results`` as the deterministic text block handed to the LLM."""
    if not results:
        return NO_CHECKS_MESSAGE

    blocks: list[str] = []
    for result in results:
        if not result.executed:
            blocks.append(f"[{result.name}] SKIPPED - {result.command}: {result.output}")
        else:
            header = f"[{result.name}] exit={result.returncode} - {result.command}"
            blocks.append(f"{header}\n{result.output}".rstrip())

    passed = sum(1 for r in results if r.executed and r.returncode == 0)
    failed = sum(1 for r in results if r.executed and r.returncode != 0)
    skipped = sum(1 for r in results if not r.executed)
    blocks.append(f"Summary: {passed} passed, {failed} failed, {skipped} skipped")
    return "\n".join(blocks)


class TesterAgent:
    # Keep pytest from trying to collect this class as a test class: its name
    # starts with "Test".
    __test__ = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        # Never pass ``settings=None`` through: the client only accepts a real
        # ``Settings`` when one is given explicitly.
        self.client = client or (LLMClient() if settings is None else LLMClient(settings=settings))

    @property
    def workspace(self) -> Path:
        """Directory whose marker files decide which checks apply."""
        workspace = getattr(self.client.settings, "workspace", None)
        return Path(workspace) if workspace is not None else Path.cwd()

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            checks = await self._run_checks()
        except OSError as exc:
            return TestResult(
                passed=False,
                summary=f"Could not inspect the workspace {self.workspace}: {exc}",
                failures=[f"runner error: {exc}"],
            )

        test_output = format_check_results(checks)
        executed = [check for check in checks if check.executed]
        failed = [check for check in executed if check.returncode != 0]

        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        verdict = self._verdict(response)

        # Verified exit codes are authoritative: a hallucinated pass is ignored.
        if failed:
            return self._failed_result(failed, verdict, len(executed))
        if not executed:
            return self._unverified_result(checks)
        return self._passed_result(response, verdict, executed)

    async def _run_tests(self) -> str:
        """Backwards-compatible text-only view over the structured check run."""
        return format_check_results(await self._run_checks())

    async def _run_checks(self) -> list[CheckResult]:
        """Run every applicable check and return one ``CheckResult`` per command."""
        results: list[CheckResult] = []
        for name, command in self._plan_checks():
            results.append(await self._run_check(name, command))
        return results

    def _plan_checks(self) -> list[tuple[str, list[str]]]:
        """Map project markers to the checks that apply to this workspace."""
        workspace = self.workspace
        checks: list[tuple[str, list[str]]] = []

        if any((workspace / marker).is_file() for marker in PYTHON_MARKERS):
            checks.append(("pytest", [sys.executable, "-m", "pytest", "-q"]))
            checks.append(("ruff", ["ruff", "check", "."]))
            checks.append(("mypy", ["mypy", *_mypy_targets(workspace)]))
        if (workspace / NODE_MARKER).is_file():
            checks.append(("npm-test", ["npm", "test", "--silent"]))
        if (workspace / RUST_MARKER).is_file():
            checks.append(("cargo-test", ["cargo", "test", "-q"]))
        if (workspace / GO_MARKER).is_file():
            checks.append(("go-test", ["go", "test", "./..."]))
        return checks

    async def _run_check(self, name: str, command: Sequence[str]) -> CheckResult:
        argv = list(command)
        display = shlex.join(argv)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            # The tool simply is not installed: skip, never a silent pass.
            return CheckResult(name, display, None, f"{argv[0]!r} is not installed", skipped=True)
        except (OSError, NotImplementedError) as exc:
            # Loop/platform cannot spawn this command (e.g. no subprocess support).
            return CheckResult(name, display, None, f"cannot run {argv[0]!r}: {exc}", skipped=True)

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=CHECK_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError, asyncio.TimeoutError):
                proc.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=KILL_TIMEOUT_SECONDS)
            return CheckResult(
                name,
                display,
                TIMEOUT_RETURNCODE,
                f"timed out after {CHECK_TIMEOUT_SECONDS:.0f}s and was killed",
            )

        combined = _truncate(_decode(stdout) + _decode(stderr))
        return CheckResult(name, display, proc.returncode, combined)

    def _verdict(self, response: str) -> TestResult | None:
        """Best-effort LLM verdict; ``None`` when the reply is not usable JSON."""
        try:
            return TestResult(**parse_json_object(response))
        except ValueError:
            return None

    def _failed_result(
        self, failed: Sequence[CheckResult], verdict: TestResult | None, executed_count: int
    ) -> TestResult:
        names = ", ".join(str(check.name) for check in failed)
        verified = [
            f"{check.name} (exit {check.returncode}): {_first_line(check.output) or 'no output'}"
            for check in failed
        ]
        failures = [*verified]
        failures.extend(f for f in (verdict.failures if verdict else []) if f not in verified)

        summary = (
            f"FAILED: {len(failed)} of {executed_count} executed check(s) exited non-zero: {names}."
        )
        if verdict is not None:
            summary += f" LLM verdict (advisory, overridden by exit codes): {verdict.summary}"
        return TestResult(passed=False, summary=summary, failures=failures)

    def _unverified_result(self, checks: Sequence[CheckResult]) -> TestResult:
        skipped = [check for check in checks if not check.executed]
        detail = "; ".join(
            f"{check.name} skipped: {check.output or 'unknown reason'}" for check in skipped
        ) or NO_CHECKS_MESSAGE
        return TestResult(
            passed=False,
            summary=(
                "UNVERIFIED: no check could be executed, so this is not a pass. " + detail
            ),
            failures=[detail],
        )

    def _passed_result(
        self, response: str, verdict: TestResult | None, executed: Sequence[CheckResult]
    ) -> TestResult:
        names = ", ".join(str(check.name) for check in executed)
        if verdict is not None:
            if verdict.passed:
                return TestResult(
                    passed=True,
                    summary=f"All executed checks passed: {names}. {verdict.summary}".strip(),
                    failures=list(verdict.failures),
                )
            failures = list(verdict.failures) or ["LLM reported a failure without naming one."]
            return TestResult(
                passed=False,
                summary=f"All executed checks passed ({names}) but the LLM reported a failure: "
                f"{verdict.summary}",
                failures=failures,
            )

        # Unparseable reply: fall back to a keyword heuristic.  Safe here because
        # every executed check already exited zero.
        lowered = response.lower()
        looks_failed = any(token in lowered for token in ("fail", "error", "traceback"))
        summary = (
            f"All executed checks passed: {names}. "
            "LLM reply was not valid JSON; verdict inferred from its raw text."
        )
        if looks_failed:
            return TestResult(
                passed=False,
                summary=summary,
                failures=[f"LLM reply was not valid JSON: {response[:MAX_OUTPUT_CHARS]}"],
            )
        return TestResult(passed=True, summary=summary, failures=[])


def _mypy_targets(workspace: Path) -> list[str]:
    """Return importable top-level package directories to typecheck."""
    try:
        packages = sorted(
            entry.name
            for entry in workspace.iterdir()
            if entry.is_dir() and (entry / "__init__.py").is_file()
        )
    except OSError:
        return ["."]
    return packages or ["."]
