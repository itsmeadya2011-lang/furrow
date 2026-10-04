from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Fallback used when ``Settings.test_timeout`` is not available.
DEFAULT_TEST_TIMEOUT = 300

# Test output is fed to the LLM, so it is capped: head + tail are kept.
MAX_TEST_OUTPUT_CHARS = 8000
TEST_OUTPUT_HEAD_CHARS = 2000

_PYTEST_CONFIG_FILES = ("pytest.ini", "tox.ini", "setup.cfg")
_PYTEST_TABLE = "[tool.pytest.ini_options]"
_TEST_DIRS = ("tests", "test")


class _Runner(NamedTuple):
    """A candidate test runner plus the check that decides if it is usable."""

    name: str
    resolve: Callable[[Path], list[str] | None]


def _has_file(workspace: Path, *names: str) -> bool:
    return any((workspace / name).is_file() for name in names)


def _is_pytest_project(workspace: Path) -> bool:
    """True when the project looks like it actually has a Python test suite."""
    if _has_file(workspace, *_PYTEST_CONFIG_FILES):
        return True
    pyproject = workspace / "pyproject.toml"
    if pyproject.is_file():
        try:
            content = pyproject.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        if _PYTEST_TABLE in content:
            return True
    return any((workspace / name).is_dir() for name in _TEST_DIRS)


def _pytest_cmd(workspace: Path) -> list[str] | None:
    # A pytest binary on PATH is sufficient: pytest itself reports an empty
    # collection with a non-zero exit code, and that is a real result.
    if shutil.which("pytest"):
        return ["pytest", "-q"]
    return None


def _python_pytest_cmd(workspace: Path) -> list[str] | None:
    interpreter = shutil.which("python") or shutil.which("python3")
    if not interpreter or not _is_pytest_project(workspace):
        return None
    return [interpreter, "-m", "pytest", "-q"]


def _node_cmd(workspace: Path, tool: str, cmd: list[str]) -> list[str] | None:
    if not shutil.which(tool) or not (workspace / "package.json").is_file():
        return None
    return list(cmd)


def _npm_cmd(workspace: Path) -> list[str] | None:
    return _node_cmd(workspace, "npm", ["npm", "test", "--", "--silent"])


def _pnpm_cmd(workspace: Path) -> list[str] | None:
    return _node_cmd(workspace, "pnpm", ["pnpm", "test", "--", "--silent"])


def _yarn_cmd(workspace: Path) -> list[str] | None:
    return _node_cmd(workspace, "yarn", ["yarn", "test", "--silent"])


def _cargo_cmd(workspace: Path) -> list[str] | None:
    if not shutil.which("cargo") or not (workspace / "Cargo.toml").is_file():
        return None
    return ["cargo", "test", "-q"]


def _go_cmd(workspace: Path) -> list[str] | None:
    if not shutil.which("go") or not (workspace / "go.mod").is_file():
        return None
    return ["go", "test", "./..."]


# Ordered by preference: Python first, then JS, then Rust, then Go.
_RUNNERS: tuple[_Runner, ...] = (
    _Runner("pytest -q", _pytest_cmd),
    _Runner("python -m pytest -q", _python_pytest_cmd),
    _Runner("npm test -- --silent", _npm_cmd),
    _Runner("pnpm test -- --silent", _pnpm_cmd),
    _Runner("yarn test --silent", _yarn_cmd),
    _Runner("cargo test -q", _cargo_cmd),
    _Runner("go test ./...", _go_cmd),
)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        """Run the first *available* test runner and report what it actually did.

        A candidate stays a candidate until it is proven available (executable
        present and the project looks like it has that ecosystem's tests). Once a
        runner actually executes, its exit code is reported: a non-zero exit is a
        test failure and never a reason to fall through to the next candidate.
        """
        workspace = self._workspace()
        timeout = self._timeout()
        checked: list[str] = []
        workspace_problem = None if workspace.is_dir() else f"workspace does not exist: {workspace}"

        for runner in _RUNNERS:
            if workspace_problem:
                checked.append(f"{runner.name} ({workspace_problem})")
                continue
            cmd = runner.resolve(workspace)
            if cmd is None:
                checked.append(f"{runner.name} (not available)")
                continue
            try:
                return await self._execute(cmd, workspace, timeout)
            except (FileNotFoundError, OSError) as exc:
                # A spawn failure means "not runnable", not "tests failed".
                checked.append(f"{' '.join(cmd)} (not runnable: {type(exc).__name__}: {exc})")

        return "No test runner found.\nChecked:\n" + "\n".join(f"  - {entry}" for entry in checked)

    async def _execute(self, cmd: list[str], workspace: Path, timeout: float) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(workspace),
            env=None,  # inherit the current process environment
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._terminate(proc)
            return (
                f"$ {' '.join(cmd)} (timed out after {timeout:.0f}s)\n"
                "The test run exceeded its timeout and was killed; "
                "no output could be captured."
            )

        returncode = proc.returncode
        status = "unknown" if returncode is None else str(returncode)
        return (
            f"$ {' '.join(cmd)} (exit {status})\n"
            f"--- stdout ---\n{self._decode(stdout) or '(empty)'}\n"
            f"--- stderr ---\n{self._decode(stderr) or '(empty)'}"
        )

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        try:
            proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except (asyncio.TimeoutError, ProcessLookupError):
            return

    def _workspace(self) -> Path:
        return Path(str(self.client.settings.workspace))

    def _timeout(self) -> float:
        raw = getattr(self.client.settings, "test_timeout", DEFAULT_TEST_TIMEOUT)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return DEFAULT_TEST_TIMEOUT
        return value if value > 0 else DEFAULT_TEST_TIMEOUT

    def _decode(self, raw: bytes) -> str:
        return self._truncate(raw.decode(errors="replace").strip())

    def _truncate(self, text: str) -> str:
        if len(text) <= MAX_TEST_OUTPUT_CHARS:
            return text
        omitted = len(text) - TEST_OUTPUT_HEAD_CHARS - MAX_TEST_OUTPUT_CHARS
        return (
            f"{text[:TEST_OUTPUT_HEAD_CHARS]}\n"
            f"... [{omitted} characters truncated] ...\n"
            f"{text[-MAX_TEST_OUTPUT_CHARS:]}"
        )
