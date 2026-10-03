from __future__ import annotations

import asyncio
import re

from furrow.agents.base import BaseAgent
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import extract_json

TEST_TIMEOUT = 120
_RUNNERS = [
    ["pytest", "-q"],
    ["python", "-m", "pytest", "-q"],
    ["npm", "test", "--", "--silent"],
    ["pnpm", "test", "--", "--silent"],
    ["yarn", "test", "--silent"],
    ["cargo", "test", "-q"],
    ["go", "test", "./..."],
]
_PASS = {"pass", "passed", "ok", "okay", "success", "green"}
_FAIL = {"fail", "failed", "failure", "error", "red"}
_TESTS_RE = re.compile(r"^(?:all\s+)?tests?\s+(passed|failed)\b")


class RunnerTimeout(Exception):
    """A test runner exists but did not finish in time: a failure, not a missing runner."""


class TesterAgent(BaseAgent):
    model_key = "tester_model"

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_output = await self._run_tests()
        except RunnerTimeout as e:
            message = str(e)
            return TestResult(passed=False, summary=message, failures=[message])
        except Exception as e:  # keep a crashed runner distinguishable from failing tests
            message = f"Test runner crashed: {type(e).__name__}: {e}"
            return TestResult(passed=False, summary=message, failures=[message])

        files = sorted({f for task in tasks for f in task.files})
        under_test = f"\n\nFiles under test: {', '.join(files)}" if files else ""
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}{under_test}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.model)
        try:
            data = extract_json(response)
        except ValueError:
            data = None
        if isinstance(data, dict):
            result = _verdict(data)
            if result is not None:
                return result
        passed, reason = _fallback_verdict(response)
        summary = f"{reason}: {response.strip()}"[:2000]
        return TestResult(passed=passed, summary=summary, failures=[])

    async def _run_tests(self) -> str:
        """Run the first available test runner in the workspace. Missing runners are skipped."""
        workspace = self.workspace
        for cmd in _RUNNERS:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (FileNotFoundError, NotADirectoryError, PermissionError):
                continue  # runner not installed, or the workspace is unreachable
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TEST_TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()  # reap the child instead of leaking a zombie
                raise RunnerTimeout(
                    f"Test runner '{' '.join(cmd)}' timed out after {TEST_TIMEOUT}s in {workspace}."
                ) from None
            return stdout.decode(errors="replace") + stderr.decode(errors="replace")
        return "No test runner found."


def _verdict(data: dict) -> TestResult | None:
    """Build a TestResult from tester JSON, or None if the shape is unusable."""
    passed = _as_bool(data.get("passed"))
    failures = data.get("failures") or []
    if passed is None or not isinstance(failures, list):
        return None
    return TestResult(
        passed=passed,
        summary=str(data.get("summary") or "Tester returned no summary."),
        failures=[str(f) for f in failures],
    )


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "pass", "passed", "success", "ok"}:
            return True
        if text in {"false", "no", "fail", "failed", "failure", "error"}:
            return False
    return None


def _fallback_verdict(response: str) -> tuple[bool, str]:
    """Conservative fallback: only an explicit verdict on the first line counts."""
    first = next((line for line in response.splitlines() if line.strip()), "").strip()
    line = first.strip("*#`_ \t-:.").lower()
    if line in _PASS:
        return True, "No JSON; explicit pass marker"
    if line in _FAIL:
        return False, "No JSON; explicit fail marker"
    match = _TESTS_RE.match(line)
    if match:
        return match.group(1) == "passed", f"No JSON; explicit '{first.strip()}' marker"
    return False, "No JSON and no explicit pass/fail marker; inconclusive"
