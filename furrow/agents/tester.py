from __future__ import annotations

import asyncio
import json
import logging
import shlex
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from furrow.agents.parsing import parse_json_response
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class TestRun:
    """Result of executing the project's test command.

    Attributes:
        output: Combined stdout/stderr of the test command (may include a
            timeout notice when the run exceeded its budget).
        returncode: The process exit code, or ``None`` when the process did
            not run (no runner found) or was killed by a timeout.
        command: The shell command that was executed, or ``None`` when no
            runner was available.
    """

    # Not a pytest test class: pytest collects any imported name starting with
    # "Test", and this dataclass has a generated __init__, which makes pytest
    # emit a PytestCollectionWarning. Do NOT "clean this up" -- the name is
    # public API, so it stays; __test__ = False is how pytest is told to skip it.
    __test__ = False

    output: str
    returncode: int | None
    command: str | None


class TesterAgent:
    # Same reason as TestRun above: pytest collects any imported Test* name, and
    # this agent class has an __init__ constructor, so without this marker every
    # module importing TesterAgent gets a PytestCollectionWarning. This is a
    # production agent, not a test.
    __test__ = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_run = await self._run_tests()
        exit_code = test_run.returncode

        # The process exit code is the AUTHORITATIVE pass/fail signal; the LLM
        # is only asked for a human-readable summary and failure enumeration.
        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n"
            f"Command: {test_run.command or 'none'}\n"
            f"Exit code: {exit_code}\n\n"
            f"Test output:\n{test_run.output}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = parse_json_response(response)
            result = TestResult(**data)
            result.passed = exit_code == 0
            return result
        except (json.JSONDecodeError, ValueError):
            # Non-JSON fallback path: keep the LLM prose as a summary, but the
            # exit code still decides pass/fail authoritatively.
            return TestResult(
                passed=exit_code == 0,
                summary=response,
                failures=[],
            )

    async def _run_tests(self) -> TestRun:
        settings = self.client.settings

        # An explicit override command always wins and is split with shlex so
        # shell quoting is respected. ``test_command``/``test_timeout`` are
        # always present on ``Settings``; ``test_command`` defaults to None,
        # which is what selects auto-detection below.
        if settings.test_command:
            cmd = shlex.split(settings.test_command)
            return await self._exec(cmd, settings.test_timeout, settings.workspace)

        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
            if not shutil.which(cmd[0]):
                continue
            return await self._exec(cmd, 120, settings.workspace)
        return TestRun(output="No test runner found.", returncode=None, command=None)

    async def _exec(self, cmd: list[str], timeout: int, workspace) -> TestRun:
        command = " ".join(shlex.quote(part) for part in cmd)
        cwd = str(workspace) if workspace is not None else None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except FileNotFoundError:
            return TestRun(
                output=f"Runner not found: {cmd[0]}", returncode=None, command=command
            )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            except (asyncio.TimeoutError, OSError) as exc:
                # Only a timeout or a closed/broken pipe is expected here, since
                # the process was just killed. Log the cause rather than hiding
                # it; the exit code is already unknown on this path.
                logger.debug(
                    "Draining test process %s after timeout failed: %r", command, exc
                )
                stdout, stderr = b"", b""
            output = _decode(stdout) + _decode(stderr) + f"\n[timed out after {timeout}s]"
            return TestRun(output=output, returncode=None, command=command)

        output = _decode(stdout) + _decode(stderr)
        return TestRun(output=output, returncode=proc.returncode, command=command)


def _decode(data: bytes | None) -> str:
    return (data or b"").decode(errors="replace")
