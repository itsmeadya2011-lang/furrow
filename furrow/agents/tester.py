from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    from furrow.config import Settings

#: Hard ceiling for a single test-runner invocation before it is killed.
TEST_TIMEOUT_SECONDS = 120

#: Test runners to try, in order.  The first one that actually exists wins.
TEST_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("npm", "test", "--", "--silent"),
    ("pnpm", "test", "--", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)

#: Errors that mean "this runner is not installed here" - the only reason to
#: fall through to the next candidate.  Anything else is a real problem.
MISSING_EXECUTABLE_ERRORS = (FileNotFoundError, NotADirectoryError, PermissionError)

NO_TEST_RUNNER_MESSAGE = "No test runner found."

_EXIT_CODE_RE = re.compile(r"^exit code: (-?\d+)", re.MULTILINE)
_TIMEOUT_RE = re.compile(r"^timed out after ([0-9.]+)s", re.MULTILINE)


def _render(command: Sequence[str]) -> str:
    return " ".join(command)


def _header(command: Sequence[str], returncode: int) -> str:
    """Build the ``$ cmd`` / ``exit code:`` prefix prefixed to runner output."""
    outcome = "passed" if returncode == 0 else "FAILED"
    return f"$ {_render(command)}\nexit code: {returncode} ({outcome})"


def _workspace_dir(settings: Settings) -> Path:
    workspace = getattr(settings, "workspace", None)
    return Path(workspace) if workspace else Path.cwd()


def _exit_code_hint(output: str) -> str:
    """One-line pass/fail summary distilled from annotated runner output."""
    timed_out = _TIMEOUT_RE.search(output)
    if timed_out is not None:
        return f"test runner timed out after {timed_out.group(1)}s and was killed"
    match = _EXIT_CODE_RE.search(output)
    if match is None:
        return ""
    code = match.group(1)
    outcome = "passed" if code == "0" else "FAILED"
    return f"test runner exited with code {code} ({outcome})"


def _with_hint(hint: str, text: str) -> str:
    return f"{hint}\n{text}" if hint else text


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_output = await self._run_tests()
        except Exception as exc:  # noqa: BLE001 - deliberate isolation boundary
            message = f"Failed to run the test suite: {exc}"
            return TestResult(passed=False, summary=message, failures=[message])

        hint = _exit_code_hint(test_output)
        sections = [TESTER_PROMPT, f"Goal: {goal}"]
        if hint:
            sections.append(f"Runner status: {hint}")
        sections.append(f"Test output:\n{test_output}")
        prompt = "\n\n".join(sections) + "\n"

        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        def heuristic() -> TestResult:
            return TestResult(
                passed="passed" in response.lower(),
                summary=_with_hint(hint, response),
                failures=[],
            )

        try:
            data = extract_json(response)
            result = TestResult(**data)
        except (TypeError, ValueError):
            # Unparseable or schema-violating LLM output: fall back to the crude
            # keyword heuristic instead of failing the whole cycle.
            return heuristic()
        return TestResult(
            passed=result.passed,
            summary=_with_hint(hint, result.summary),
            failures=result.failures,
        )

    async def _run_tests(self) -> str:
        """Run the first available test runner and return its annotated output.

        Runs inside the configured workspace so the runner sees the project's own
        tests.  Only a missing executable moves on to the next candidate; any other
        error propagates because it indicates something actually broken.  The output
        is prefixed with the command and its return code so pass/fail is visible to
        both the LLM and the human logs.
        """
        cwd = _workspace_dir(self.client.settings)
        for command in TEST_COMMANDS:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            except MISSING_EXECUTABLE_ERRORS:
                continue

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=TEST_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()  # reap the child, otherwise it lingers as a zombie
                return (
                    f"$ {_render(command)}\n"
                    f"timed out after {TEST_TIMEOUT_SECONDS}s; process killed\n"
                )

            return (
                f"{_header(command, proc.returncode)}\n"
                f"{stdout.decode(errors='replace')}{stderr.decode(errors='replace')}"
            )
        return NO_TEST_RUNNER_MESSAGE