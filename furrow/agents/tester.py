from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

TestStatus = Literal["passed", "failed", "timeout", "no_runner"]

#: Exceptions that only mean "this executable is not usable on this machine".
_PROBE_ERRORS = (FileNotFoundError, NotADirectoryError, PermissionError)

_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*\n?|\n?```\s*$")
_PASSED_JSON_RE = re.compile(r'"passed"\s*:\s*true', re.IGNORECASE)
_PASSED_TOKEN_RE = re.compile(r"(?<![\w\"'])PASSED(?![\w\"'])")

_DEFAULT_TIMEOUT = 120.0


@dataclass
class TestRun:
    """Outcome of probing the project's test runner.

    ``status`` is ``"passed"`` / ``"failed"`` when a runner executed to
    completion, ``"timeout"`` when it exceeded the deadline (process killed and
    reaped), and ``"no_runner"`` when no candidate runner could be started.
    """

    status: TestStatus
    passed: bool = False
    output: str = ""
    returncode: int | None = None
    command: list[str] | None = None
    tried: list[str] = field(default_factory=list)


def _decode(stdout: bytes | str | None, stderr: bytes | str | None) -> str:
    def part(value: bytes | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    return part(stdout) + part(stderr)


def _strip_fences(text: str) -> str:
    """Remove surrounding markdown code fences (``` or ```json) from ``text``."""
    return _FENCE_RE.sub("", text).strip()


def _render_tasks(tasks: list[TaskModel]) -> str:
    if not tasks:
        return "- (no tasks were planned)"
    return "\n".join(f"- {task.id}: {task.description}" for task in tasks)


class TesterAgent:
    # Not a pytest test class; the `Test*` prefix is part of the domain name.
    __test__ = False

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self.timeout = timeout

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        run = await self._run_tests(timeout=self.timeout)

        if run.status == "no_runner":
            tried = ", ".join(run.tried) or "none"
            summary = f"No test runner found; tried: {tried}"
            return TestResult(
                passed=False,
                summary=summary,
                failures=[summary, "No test runner found, so the changes could not be verified."],
            )

        detail = f"returncode={run.returncode}" if run.returncode is not None else "returncode=None"
        command = " ".join(run.command) if run.command else "unknown"
        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Tasks implemented:\n{_render_tasks(tasks)}\n\n"
            f"Test runner: {command}\n"
            f"Test runner status: {run.status} ({detail})\n\n"
            f"Test output:\n{run.output}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        return self._verdict(response, run)

    def _verdict(self, response: str, run: TestRun) -> TestResult:
        raw = response or ""
        try:
            data = json.loads(_strip_fences(raw))
        except (json.JSONDecodeError, ValueError):
            data = None

        if isinstance(data, dict):
            passed = bool(data.get("passed", False))
            summary = str(data.get("summary", "")).strip() or raw.strip() or "No summary provided."
            failures = [str(item) for item in (data.get("failures") or [])]
            # A judge that says "passed" while the runner failed never passes.
            if run.status != "passed":
                passed = False
            if not passed and not failures:
                failures = [f"Test runner {run.status} ({run.output.strip()[:2000]})"]
            return TestResult(passed=passed, summary=summary, failures=failures)

        # Unparseable output: only trust an unambiguous, structured signal.
        passed = bool(_PASSED_JSON_RE.search(raw) or _PASSED_TOKEN_RE.search(raw))
        if run.status != "passed":
            passed = False
        summary = (
            f"Unparsable judge response: {raw.strip()}" if raw.strip() else "Empty judge response."
        )
        return TestResult(
            passed=passed,
            summary=summary,
            failures=[] if passed else [f"Test runner {run.status}: {run.output.strip()[:2000]}"],
        )

    async def _run_tests(self, timeout: float = _DEFAULT_TIMEOUT) -> TestRun:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        tried: list[str] = []
        for cmd in candidates:
            label = " ".join(cmd)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except _PROBE_ERRORS:
                tried.append(label)
                continue

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()  # reap: no zombie left behind
                partial = await self._drain(proc)
                return TestRun(
                    status="timeout",
                    passed=False,
                    output=partial,
                    returncode=proc.returncode,
                    command=cmd,
                    tried=tried,
                )

            output = _decode(stdout, stderr)
            returncode = proc.returncode
            return TestRun(
                status="passed" if returncode == 0 else "failed",
                passed=returncode == 0,
                output=output,
                returncode=returncode,
                command=cmd,
                tried=tried,
            )

        return TestRun(status="no_runner", passed=False, output="", tried=tried)

    @staticmethod
    async def _drain(proc: object) -> str:
        """Best-effort collection of output produced before the timeout.

        Called only after the process was killed and reaped; failures here must
        never mask the timeout verdict, so they degrade to an empty string.
        """
        communicate = getattr(proc, "communicate", None)
        if communicate is None:
            return ""
        try:
            stdout, stderr = await asyncio.wait_for(communicate(), timeout=5.0)
        except asyncio.CancelledError:
            raise
        except Exception:  # best effort output recovery only
            return ""
        return _decode(stdout, stderr)
