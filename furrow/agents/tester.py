from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from furrow.agents.planner import extract_json_object
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

#: Test runners tried in order; the first one that actually exists is used.
TEST_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("npm", "test", "--", "--silent"),
    ("pnpm", "test", "--", "--silent"),
    ("yarn", "test", "--silent"),
    ("cargo", "test", "-q"),
    ("go", "test", "./..."),
)

NO_RUNNER_OUTPUT = "No test runner found."


@dataclass
class TestRun:
    """Outcome of a single test-runner invocation."""

    ran: bool
    returncode: int
    output: str
    timed_out: bool = False


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1", "passed", "pass"):
            return True
        if lowered in ("false", "no", "0", "failed", "fail"):
            return False
    return default


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


class TesterAgent:
    #: Not a pytest test class (the name starts with ``Test``).
    __test__ = False

    #: Seconds a single test-runner invocation may take before it is killed.
    test_timeout: float = 120.0

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        timeout: float | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        if timeout is not None:
            self.test_timeout = float(timeout)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_run = await self._run_tests()
        except Exception as e:  # genuine execution error -> report as a failure
            message = f"{type(e).__name__}: {e}"
            return TestResult(
                passed=False,
                summary=f"Test execution failed: {message}",
                failures=[message],
            )

        output = test_run.output or NO_RUNNER_OUTPUT
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        # The exit code is the source of truth whenever a runner actually ran.
        exit_code_passed = test_run.ran and test_run.returncode == 0

        try:
            data = extract_json_object(response)
        except ValueError:
            return TestResult(
                passed=exit_code_passed,
                summary=response.strip() or output,
                failures=[],
            )

        summary = data.get("summary")
        summary = summary.strip() if isinstance(summary, str) else ""
        raw_failures = data.get("failures")
        failures = [str(f) for f in raw_failures] if isinstance(raw_failures, list) else []

        passed = exit_code_passed if test_run.ran else _coerce_bool(data.get("passed"))
        return TestResult(
            passed=passed,
            summary=summary or response.strip() or output,
            failures=failures,
        )

    async def _run_tests(self) -> TestRun:
        timed_out: TestRun | None = None
        for cmd in TEST_COMMANDS:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                # Runner not installed on this machine: try the next candidate.
                continue

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.test_timeout
                )
            except (asyncio.TimeoutError, TimeoutError):
                proc.kill()
                await proc.wait()
                timed_out = TestRun(
                    ran=True,
                    returncode=proc.returncode if proc.returncode is not None else -1,
                    output=f"Test run timed out after {self.test_timeout}s: {' '.join(cmd)}",
                    timed_out=True,
                )
                continue

            return TestRun(
                ran=True,
                returncode=proc.returncode if proc.returncode is not None else -1,
                output=_decode(stdout) + _decode(stderr),
            )

        if timed_out is not None:
            return timed_out
        return TestRun(ran=False, returncode=-1, output=NO_RUNNER_OUTPUT)