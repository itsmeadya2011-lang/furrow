from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


@dataclass
class _TestRunResult:
    output: str
    stdout: str
    stderr: str
    returncode: int
    runner_found: bool
    timed_out: bool = False


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        result = await self._run_tests()

        if not result.runner_found:
            if result.timed_out:
                return TestResult(
                    passed=False,
                    summary="All test runners timed out.",
                    failures=["All test runners timed out."],
                )
            return TestResult(
                passed=False,
                summary="No test runner found. Install pytest, npm, pnpm, yarn, cargo, or go.",
                failures=["No test runner found."],
            )

        if result.returncode == 0:
            return TestResult(passed=True, summary="All tests passed.", failures=[])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{result.output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        data = self._parse_test_result(response)

        return TestResult(
            passed=False,
            summary=data["summary"],
            failures=data["failures"] or [data["summary"] or "Tests failed."],
        )

    def _parse_test_result(self, response: str) -> dict[str, Any]:
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            text = match.group(0) if match else response
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                passed = data.get("passed", False)
                if not isinstance(passed, bool):
                    passed = False
                summary = str(data.get("summary", ""))
                failures = data.get("failures", [])
                if not isinstance(failures, list):
                    failures = []
                failures = [str(f) for f in failures]
                return {
                    "passed": passed,
                    "summary": summary,
                    "failures": failures,
                }
        except (json.JSONDecodeError, ValueError):
            pass
        return {
            "passed": False,
            "summary": response,
            "failures": [response],
        }

    async def _run_tests(self) -> _TestRunResult:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        runner_found = False
        timed_out = False
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except OSError:
                continue
            runner_found = True
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=120)
                return _TestRunResult(
                    output=stdout_bytes.decode(errors="replace") + stderr_bytes.decode(errors="replace"),
                    stdout=stdout_bytes.decode(errors="replace"),
                    stderr=stderr_bytes.decode(errors="replace"),
                    returncode=proc.returncode,
                    runner_found=True,
                    timed_out=False,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                timed_out = True
                continue

        return _TestRunResult(
            output="All test runners timed out." if timed_out else "No test runner found.",
            stdout="",
            stderr="",
            returncode=-1,
            runner_found=False,
            timed_out=timed_out,
        )
