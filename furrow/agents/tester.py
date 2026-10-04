from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.planner import extract_json
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_FAILURES = 20
MAX_FAILURE_CHARS = 300
OUTPUT_TAIL_CHARS = 8_000
SUMMARY_TAIL_CHARS = 600
DEFAULT_TEST_TIMEOUT = 300
UNKNOWN_EXIT_CODE = -1

PYTHON_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "pytest.ini",
    "tox.ini",
    "requirements.txt",
)
NODE_MARKERS = ("package.json",)
RUST_MARKERS = ("Cargo.toml",)
GO_MARKERS = ("go.mod",)

_FAILURE_MARKERS = (
    "FAILED",
    "FAIL ",
    "--- FAILURES ---",
    "ERROR",
    "AssertionError",
    "Traceback",
    "error:",
    "not ok",
    "E   ",
    "✗",
    "✘",
)
_SUMMARY_RE = re.compile(r"^\s*\d+\s+(failed|failing|error|errors|passed|skipped|total)", re.I)


def _one_line(text: str, width: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > width:
        return collapsed[:width].rstrip() + "..."
    return collapsed


def _tail(text: str, width: int) -> str:
    return text[-width:] if len(text) > width else text


def _decode(chunk: bytes | None) -> str:
    return (chunk or b"").decode("utf-8", errors="replace")


def _collect_failures(output: str, limit: int = MAX_FAILURES) -> list[str]:
    """Pull failure-shaped lines out of test output without trusting any verdict."""
    failures: list[str] = []
    in_traceback = False
    for line in output.splitlines():
        if not line.strip():
            in_traceback = False
            continue
        keep = any(marker in line for marker in _FAILURE_MARKERS)
        keep = keep or _SUMMARY_RE.match(line) is not None
        if "Traceback" in line:
            in_traceback = True
            keep = True
        elif in_traceback and line[:1].isspace():
            keep = True
        if keep:
            failures.append(_one_line(line, MAX_FAILURE_CHARS))
            if len(failures) >= limit:
                break
    return failures


class TesterAgent:
    """Runs the workspace's own test command and reports what actually happened.

    The verdict comes from the process exit code. The LLM is only asked to phrase
    that verdict; it can never change it.
    """

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self._command: list[str] | None = None
        self._returncode: int | None = None

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            outcome = await self._run_tests()
        except TimeoutError as exc:
            detail = str(exc)
            return TestResult(passed=False, summary=detail, failures=[detail])
        except (OSError, RuntimeError, ValueError) as exc:
            detail = f"Could not run the test suite: {exc}"
            return TestResult(passed=False, summary=detail, failures=[detail])

        if outcome is None:
            detail = (
                "No test runner detected: no executable pytest/npm/pnpm/yarn/cargo/go "
                f"command matching a recognised project layout in {self._workspace()}"
            )
            return TestResult(passed=False, summary=detail, failures=[detail])

        output, returncode = outcome
        self._returncode = returncode
        command = " ".join(self._command or [])
        failures = _collect_failures(output) if returncode != 0 else []
        if returncode != 0 and not failures:
            tail = _tail(output.strip(), SUMMARY_TAIL_CHARS)
            failures = [f"{command} exited {returncode}: {tail}"]
        decided = TestResult(
            passed=returncode == 0,
            summary=(
                f"`{command}` exited {returncode}. "
                f"{_tail(output.strip(), SUMMARY_TAIL_CHARS)}"
            ).strip(),
            failures=failures,
        )
        return await self._summarize(goal, tasks, decided, output)

    def _workspace(self) -> Path:
        raw = getattr(self.client.settings, "workspace", None) or Path.cwd()
        path = Path(raw).expanduser()
        try:
            return path.resolve()
        except OSError:
            return path.absolute()

    def _timeout(self) -> float:
        value = getattr(self.client.settings, "test_timeout", DEFAULT_TEST_TIMEOUT)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(DEFAULT_TEST_TIMEOUT)

    def _candidates(self) -> list[list[str]]:
        """Runner commands for the ecosystems this workspace actually contains.

        Restricting by manifest keeps a stray pytest on PATH from hijacking a
        JavaScript, Rust or Go project.
        """
        workspace = self._workspace()

        def has(markers: tuple[str, ...]) -> bool:
            return any((workspace / marker).is_file() for marker in markers)

        candidates: list[list[str]] = []
        if has(PYTHON_MARKERS):
            candidates.append(["pytest", "-q"])
            candidates.append([sys.executable, "-m", "pytest", "-q"])
        if has(NODE_MARKERS):
            candidates.append(["npm", "test", "--", "--silent"])
            candidates.append(["pnpm", "test", "--", "--silent"])
            candidates.append(["yarn", "test", "--silent"])
        if has(RUST_MARKERS):
            candidates.append(["cargo", "test", "-q"])
        if has(GO_MARKERS):
            candidates.append(["go", "test", "./..."])
        if not candidates:
            candidates.append(["pytest", "-q"])
            candidates.append([sys.executable, "-m", "pytest", "-q"])
        return candidates

    async def _detect_runner(self) -> list[str] | None:
        """Return the first candidate that is executable here, without spawning."""
        for command in self._candidates():
            if self._executable(command):
                return command
        return None

    @staticmethod
    def _executable(command: list[str]) -> bool:
        head = command[0]
        if head == sys.executable:
            return importlib.util.find_spec("pytest") is not None
        return shutil.which(head) is not None

    async def _run_tests(self) -> tuple[str, int] | None:
        """Run the detected command in the workspace, returning (output, returncode)."""
        command = await self._detect_runner()
        if command is None:
            return None
        self._command = command
        timeout = self._timeout()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self._workspace()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to start {' '.join(command)}: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            process.kill()
            await _reap(process)
            message = f"Test run timed out after {timeout:.0f}s: {' '.join(command)}"
            raise TimeoutError(message) from exc
        output = _decode(stdout) + _decode(stderr)
        code = process.returncode if process.returncode is not None else UNKNOWN_EXIT_CODE
        return output, code

    async def _summarize(
        self,
        goal: str,
        tasks: list[TaskModel],
        decided: TestResult,
        output: str,
    ) -> TestResult:
        """Let the LLM phrase the verdict, never overrule it."""
        task_lines = "\n".join(f"- {task.id}: {task.description}" for task in tasks[:10])
        prompt = (
            f"Goal: {goal}\n\n"
            f"Tasks attempted:\n{task_lines or '(none)'}\n\n"
            f"Command: {' '.join(self._command or [])}\n"
            f"Exit code: {self._exit_code()}\n"
            "VERDICT (already decided from the exit code, copy it exactly): "
            f"passed={str(decided.passed).lower()}\n\n"
            f"Test output (tail):\n{_tail(output, OUTPUT_TAIL_CHARS)}\n\n"
            "Summarise this result as JSON now."
        )
        try:
            response = await self.client.complete(
                prompt,
                system=TESTER_PROMPT,
                model=getattr(self.client.settings, "tester_model", None),
            )
        except Exception:  # noqa: BLE001 - the exit-code verdict stands on its own
            return decided
        try:
            data: Any = extract_json(response)
        except ValueError:
            return decided
        if not isinstance(data, dict):
            return decided
        summary = data.get("summary")
        failures = data.get("failures")
        if not isinstance(summary, str) or not summary.strip():
            summary = decided.summary
        if isinstance(failures, list):
            listed = [
                _one_line(str(item), MAX_FAILURE_CHARS) for item in failures if str(item).strip()
            ]
            failures = listed or decided.failures
        else:
            failures = decided.failures
        return TestResult(passed=decided.passed, summary=summary.strip(), failures=failures)

    def _exit_code(self) -> int:
        return self._returncode if self._returncode is not None else UNKNOWN_EXIT_CODE


async def _reap(process: asyncio.subprocess.Process) -> None:
    """Wait for a killed child so it does not linger as a zombie."""
    with contextlib.suppress(ProcessLookupError):
        await process.wait()
